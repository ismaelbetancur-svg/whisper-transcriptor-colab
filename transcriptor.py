import os
import sys
import time
import subprocess

# --- 1. AUTO-INSTALACIÓN DE DEPENDENCIAS ---
# Verifica si whisper y tqdm están instalados; si no, los instala solo.
def install_dependencies():
    print("🚀 Verificando dependencias del sistema...")
    packages = ["openai-whisper", "tqdm"]
    installed = False
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            print(f"⬇️ Instalando {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package])
            installed = True
    if installed:
        print("✅ Dependencias instaladas. Iniciando...")
    else:
        print("✅ Dependencias listas.")

install_dependencies()

# --- IMPORTACIONES ---
import torch
import difflib
import re
import shutil
import whisper
from whisper.utils import get_writer
from tqdm import tqdm
from google.colab import drive

# --- INTERFAZ DE USUARIO ---
print("\n" + "="*50)
print("   🎙️  TRANSCRIPTOR WHISPER PRO (AUTOMATIZADO)  🎙️")
print("="*50)

# 2. Montar Drive
print("\n📂 Conectando con Google Drive...")
drive.mount('/content/drive')

# --- 3. CONFIGURACIÓN INTERACTIVA ---
print("\n" + "-"*30)

# PREGUNTA 1: Carpeta (Con valor por defecto)
input_folder = input("📂 Nombre de la carpeta en Drive (Enter para usar 'videos'): ").strip()
if not input_folder:
    input_folder = "videos"  # <--- VALOR POR DEFECTO
print(f"   👉 Usando carpeta: '{input_folder}'")

# PREGUNTA 2: Archivo
nombre_archivo = input("🎵 Nombre del archivo (ej: 1.mp3): ").strip()
if not nombre_archivo:
    sys.exit("❌ Error: Debes escribir el nombre del archivo.")

# --- DEFINICIÓN DE RUTAS ---
base_path = "/content/drive/My Drive"
target_dir = os.path.join(base_path, input_folder) # <--- Aquí se guardará el SRT también
input_file = os.path.join(target_dir, nombre_archivo)

# Validaciones
if not os.path.exists(target_dir):
    sys.exit(f"❌ ERROR: La carpeta '{input_folder}' no existe en tu Google Drive.")
if not os.path.exists(input_file):
    sys.exit(f"❌ ERROR: No encuentro el archivo '{nombre_archivo}' dentro de '{input_folder}'.")

print(f"✅ Archivo encontrado: {input_file}")
print(f"✅ El SRT final se guardará en: {target_dir}") # <--- Comentario para el usuario

# --- 4. CARGAR MODELO ---
print("\n🧠 Cargando modelo Whisper Large-v3 en GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = whisper.load_model("large-v3", device=device)
except Exception as e:
    sys.exit(f"❌ ERROR al cargar modelo: {e}")

# --- 5. TRANSCRIPCIÓN ---
print(f"\n🎙️ Transcribiendo (Esto puede tomar tiempo)...")

# Prompt Anti-Alucinaciones
texto_prompt = "Hello. This is a clear transcript with no repetitions, no hesitation, and correct punctuation."

transcription_options = {
    "language": "en",
    "verbose": False,
    "word_timestamps": True,
    "fp16": True,
    "temperature": 0.0,
    "best_of": 5,
    "beam_size": 5,
    "condition_on_previous_text": False,
    "initial_prompt": texto_prompt,
    "no_speech_threshold": 0.4,
    "logprob_threshold": -1.0
}

start_time = time.time()
result = model.transcribe(input_file, **transcription_options)
end_time = time.time()

if not result or 'segments' not in result or len(result['segments']) == 0:
    sys.exit("❌ ERROR CRÍTICO: Whisper no generó texto.")

print(f"✅ Transcripción base lista ({(end_time - start_time)/60:.2f} min).")

# --- 6. LIMPIEZA INTELIGENTE ---
print(f"\n🧹 Limpiando repeticiones, prompt fantasma y ajustando tiempos...")

def clean_text_content(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\(.*?\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_similar(a, b, threshold=0.9):
    return difflib.SequenceMatcher(None, a, b).ratio() > threshold

def process_segments_safe(segments):
    cleaned = []
    last_text = ""
    last_end_time = 0.0
    dropped_count = 0

    # Frases que queremos eliminar si la IA las transcribe por error
    frases_prohibidas = [
        "This is a clear transcript", "with no repetitions",
        "no hesitation", "correct punctuation",
        "Hello. This is a clear transcript"
    ]

    for segment in tqdm(segments, desc="Procesando líneas", unit="linea"):
        try:
            text = clean_text_content(segment['text'])
            text_lower = text.lower()

            if not text: continue

            # Filtro Anti-Prompt
            es_prompt = False
            for frase in frases_prohibidas:
                if frase.lower() in text_lower:
                    es_prompt = True; break
            if es_prompt: dropped_count += 1; continue

            # Filtros Estándar (Basura en silencios)
            if text_lower in ["thank you.", "copyright", "bye."] and (segment['end'] - segment['start'] < 2.0):
                dropped_count += 1; continue
            
            # Filtro de Repetición (Fuzzy)
            if is_similar(text_lower, last_text.lower()):
                dropped_count += 1; continue

            # Sync Fix (Evitar superposición de tiempos)
            start_time = segment['start']
            if start_time < last_end_time: start_time = last_end_time + 0.05
            if start_time >= segment['end']: continue

            segment['start'] = start_time
            segment['text'] = text
            cleaned.append(segment)
            last_text = text
            last_end_time = segment['end']
        except: continue
    
    print(f"   📉 Líneas eliminadas: {dropped_count}")
    return cleaned

result_cleaned = result.copy()
result_cleaned['segments'] = process_segments_safe(result['segments'])

# --- 7. GUARDAR ---
print(f"\n💾 Generando SRT Final (Sin etiquetas raras)...")

# highlight_words = False para evitar el <u></u> y tener texto plano
writer_options = {"highlight_words": False, "max_line_width": 50, "max_line_count": 2}

nombre_final = os.path.splitext(nombre_archivo)[0] + "_FINAL"
# Usamos ruta falsa para engañar al writer y forzar nombre
fake_input_path = os.path.join(target_dir, nombre_final + ".mp3")

try:
    srt_writer = get_writer("srt", target_dir)
    srt_writer(result_cleaned, fake_input_path, writer_options)
except Exception: pass

# Mover y confirmar ubicación
archivo_generado = os.path.join(target_dir, nombre_final + ".srt")
local_file = nombre_final + ".srt"

if not os.path.exists(archivo_generado) and os.path.exists(local_file):
    shutil.move(local_file, archivo_generado)

print("\n" + "="*40)
if os.path.exists(archivo_generado):
    print("🎉 ¡PROCESO EXITOSO!")
    print(f"📂 Archivo guardado en la misma carpeta: {archivo_generado}")
else:
    print("❌ ERROR: No encuentro el archivo final. Revisa los archivos temporales.")
print("="*40)