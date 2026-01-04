import os
import sys
import time
import subprocess
import glob

# --- 1. AUTO-INSTALACIÓN ---
def install_dependencies():
    print("🚀 Verificando dependencias...")
    packages = ["openai-whisper", "tqdm"]
    installed = False
    for package in packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "-U", package])
            installed = True
    if installed: print("✅ Dependencias listas.")

install_dependencies()

import torch
import difflib
import re
import shutil
import whisper
from whisper.utils import get_writer
from tqdm import tqdm

# --- INTERFAZ ---
print("\n" + "="*60)
print("   🎙️  TRANSCRIPTOR WHISPER PRO (SELECCIÓN MÚLTIPLE + TXT)  🎙️")
print("="*60)

# --- 2. VERIFICACIÓN DE DRIVE ---
if not os.path.exists('/content/drive'):
    print("\n❌ ERROR: Google Drive no está montado.")
    print("   👉 Ejecuta primero: from google.colab import drive; drive.mount('/content/drive')")
    sys.exit(1)

# --- 3. CONFIGURACIÓN ---
print("\n--- 📂 RUTAS ---")
input_folder = input("   📂 Carpeta en Drive (Enter para 'videos'): ").strip() or "videos"
base_path = "/content/drive/My Drive"
target_dir = os.path.join(base_path, input_folder)

if not os.path.exists(target_dir):
    sys.exit(f"❌ Error: La carpeta '{input_folder}' no existe en tu Drive.")

print(f"   📂 Trabajando en: {target_dir}")

# --- SELECCIÓN INTELIGENTE DE ARCHIVOS ---
print("\n--- 🎵 SELECCIÓN DE ARCHIVOS ---")
print("   Opciones:")
print("   1. 'TODOS' (Procesa todo el contenido de la carpeta)")
print("   2. 'video1.mp4' (Un solo archivo)")
print("   3. '1.mp3, 2.wav, 3.m4a' (Varios archivos separados por coma)")
user_input = input("   👉 ¿Qué procesamos?: ").strip()

files_to_process = []

if not user_input:
    sys.exit("❌ Error: No escribiste nada.")

if user_input.upper() == "TODOS":
    # Modo masivo
    types = ('*.mp3', '*.wav', '*.m4a', '*.mp4', '*.mpeg', '*.mov', '*.mkv')
    for t in types:
        files_to_process.extend(glob.glob(os.path.join(target_dir, t)))
    print(f"   📦 Modo Masivo: Se encontraron {len(files_to_process)} archivos.")
else:
    # Modo lista (uno o varios)
    nombres = [x.strip() for x in user_input.split(',')]
    for nombre in nombres:
        ruta_completa = os.path.join(target_dir, nombre)
        if os.path.exists(ruta_completa):
            files_to_process.append(ruta_completa)
        else:
            print(f"   ⚠️ Advertencia: No se encontró '{nombre}', se omitirá.")

if not files_to_process:
    sys.exit("❌ No hay archivos válidos para procesar.")

print(f"   ✅ Lista final: {len(files_to_process)} archivos en cola.")

# --- CONFIGURACIÓN MODELO ---
print("\n--- 🧠 CONFIGURACIÓN ---")
model_name = input("   👉 Modelo (Enter='large-v3'): ").strip() or "large-v3"
language = input("   🌐 Idioma (es, en, fr) [Enter='en']: ").strip() or "en"

print(f"\n🚀 Cargando modelo '{model_name}' en GPU...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = whisper.load_model(model_name, device=device)
except Exception as e: sys.exit(f"❌ Error cargando modelo: {e}")

# --- 4. PROCESAMIENTO ---
# Funciones de limpieza
def clean_text(text): return re.sub(r'\[.*?\]|\(.*?\)', '', text).strip()
def is_similar(a, b): return difflib.SequenceMatcher(None, a, b).ratio() > 0.9

# Prompt
if language == "en": prompt = "Hello. This is a clear transcript with no repetitions."
elif language == "es": prompt = "Hola. Transcripción clara, sin repeticiones."
else: prompt = "Clear transcript."

transcription_opts = {
    "language": language, "verbose": False, "word_timestamps": True,
    "fp16": True, "temperature": 0.0, "best_of": 5, "beam_size": 5,
    "condition_on_previous_text": False, "initial_prompt": prompt,
    "no_speech_threshold": 0.4, "logprob_threshold": -1.0
}

writer_opts = {"highlight_words": False, "max_line_width": 50, "max_line_count": 2}

print("\n" + "="*60)
for i, input_file in enumerate(files_to_process):
    filename = os.path.basename(input_file)
    print(f"▶️  [{i+1}/{len(files_to_process)}] Procesando: {filename}")
    
    start_t = time.time()
    try:
        result = model.transcribe(input_file, **transcription_opts)
    except Exception as e:
        print(f"   ❌ Error con {filename}: {e}")
        continue

    if not result or not result['segments']:
        print("   ⚠️ Audio vacío.")
        continue

    # Limpieza
    cleaned = []
    last_txt = ""; last_end = 0.0
    ban = ["this is a clear transcript", "transcripción clara", "sin repeticiones"]

    for seg in result['segments']:
        txt = clean_text(seg['text'])
        if not txt: continue
        if any(b in txt.lower() for b in ban): continue
        if is_similar(txt.lower(), last_txt.lower()): continue
        
        start = seg['start']
        if start < last_end: start = last_end + 0.05
        if start >= seg['end']: continue
        
        seg['start'] = start; seg['text'] = txt
        cleaned.append(seg)
        last_txt = txt; last_end = seg['end']
    
    result['segments'] = cleaned

    # Guardar
    base = os.path.splitext(filename)[0] + "_FINAL"
    
    # SRT
    try:
        get_writer("srt", target_dir)(result, os.path.join(target_dir, base + ".mp3"), writer_opts)
        if os.path.exists(base + ".srt"):
            shutil.move(base + ".srt", os.path.join(target_dir, base + ".srt"))
    except: pass

    # TXT
    with open(os.path.join(target_dir, base + ".txt"), "w", encoding="utf-8") as f:
        for s in cleaned: f.write(s['text'] + " ")

    print(f"   ✅ Terminado en {(time.time()-start_t)/60:.2f} min (SRT + TXT creados).")

print("\n" + "="*60)
print(f"🎉 ¡TODO LISTO! Revisa tu carpeta '{input_folder}'.")