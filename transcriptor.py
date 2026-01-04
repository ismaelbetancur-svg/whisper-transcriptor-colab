import os
import sys
import time
import subprocess

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

# --- IMPORTACIONES ---
import torch
import difflib
import re
import shutil
import whisper
from whisper.utils import get_writer
from tqdm import tqdm

# --- INTERFAZ ---
print("\n" + "="*50)
print("   🎙️  TRANSCRIPTOR WHISPER PRO (V. GITHUB)  🎙️")
print("="*50)

# --- 2. VERIFICACIÓN DE DRIVE (CORREGIDO) ---
# Ya no intentamos montar aquí para evitar el error de Kernel.
if not os.path.exists('/content/drive'):
    print("\n❌ ERROR CRÍTICO: Google Drive no está conectado.")
    print("   👉 SOLUCIÓN: Antes de ejecutar este script, debes montar Drive en la celda de Colab.")
    sys.exit(1)
else:
    print("\n✅ Google Drive detectado correctamente.")

# --- 3. CONFIGURACIÓN ---
print("-" * 30)
input_folder = input("📂 Nombre de la carpeta en Drive (Enter para 'videos'): ").strip()
if not input_folder: input_folder = "videos"

nombre_archivo = input("🎵 Nombre del archivo (ej: 1.mp3): ").strip()
if not nombre_archivo: sys.exit("❌ Error: Nombre de archivo vacío.")

base_path = "/content/drive/My Drive"
target_dir = os.path.join(base_path, input_folder)
input_file = os.path.join(target_dir, nombre_archivo)

if not os.path.exists(target_dir): sys.exit(f"❌ Carpeta no encontrada: {target_dir}")
if not os.path.exists(input_file): sys.exit(f"❌ Archivo no encontrado: {input_file}")

print(f"✅ Procesando: {input_file}")

# --- 4. CARGAR MODELO ---
print("\n🧠 Cargando Whisper Large-v3...")
device = "cuda" if torch.cuda.is_available() else "cpu"
try:
    model = whisper.load_model("large-v3", device=device)
except Exception as e: sys.exit(f"❌ Error GPU: {e}")

# --- 5. TRANSCRIPCIÓN ---
print(f"\n🎙️ Transcribiendo...")
texto_prompt = "Hello. This is a clear transcript with no repetitions, no hesitation, and correct punctuation."

transcription_options = {
    "language": "en", "verbose": False, "word_timestamps": True,
    "fp16": True, "temperature": 0.0, "best_of": 5, "beam_size": 5,
    "condition_on_previous_text": False, "initial_prompt": texto_prompt,
    "no_speech_threshold": 0.4, "logprob_threshold": -1.0
}

start_time = time.time()
result = model.transcribe(input_file, **transcription_options)
if not result or not result['segments']: sys.exit("❌ Error: Audio vacío o ilegible.")
print(f"✅ Transcripción base lista ({(time.time()-start_time)/60:.2f} min).")

# --- 6. LIMPIEZA ---
print(f"\n🧹 Limpiando...")
def clean_text(text):
    text = re.sub(r'\[.*?\]|\(.*?\)', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_similar(a, b): return difflib.SequenceMatcher(None, a, b).ratio() > 0.9

cleaned_segments = []
last_text = ""
last_end = 0.0
dropped = 0
frases_ban = ["this is a clear transcript", "with no repetitions", "correct punctuation"]

for seg in tqdm(result['segments'], unit="linea"):
    txt = clean_text(seg['text'])
    if not txt: continue
    
    # Filtros
    if any(f in txt.lower() for f in frases_ban): dropped += 1; continue
    if txt.lower() in ["thank you.", "copyright", "bye."] and (seg['end']-seg['start'] < 2): dropped += 1; continue
    if is_similar(txt.lower(), last_text.lower()): dropped += 1; continue
    
    # Sync Fix
    start = seg['start']
    if start < last_end: start = last_end + 0.05
    if start >= seg['end']: continue
    
    seg['start'] = start; seg['text'] = txt
    cleaned_segments.append(seg)
    last_text = txt; last_end = seg['end']

print(f"   📉 Eliminadas {dropped} líneas basura.")
result['segments'] = cleaned_segments

# --- 7. GUARDAR ---
print(f"\n💾 Guardando SRT...")
writer_opts = {"highlight_words": False, "max_line_width": 50, "max_line_count": 2}
final_name = os.path.splitext(nombre_archivo)[0] + "_FINAL"
fake_path = os.path.join(target_dir, final_name + ".mp3")

try: get_writer("srt", target_dir)(result, fake_path, writer_opts)
except: pass

real_file = os.path.join(target_dir, final_name + ".srt")
temp_file = final_name + ".srt"
if not os.path.exists(real_file) and os.path.exists(temp_file):
    shutil.move(temp_file, real_file)

if os.path.exists(real_file): print(f"🎉 ÉXITO: {real_file}")
else: print("❌ Error guardando archivo.")