## 🚀 Cómo usar en Google Colab

Copia y pega este comando en una celda:

from google.colab import drive
drive.mount('/content/drive')

!wget -O run_whisper.py https://raw.githubusercontent.com/ismaelbetancur-svg/whisper-transcriptor-colab/main/transcriptor.py && python run_whisper.py
