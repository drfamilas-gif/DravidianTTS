import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

plt.rcParams["font.family"] = "Space Mono"

# ================= CONFIG =================
DATA_ROOT = "Data/data"
OUT_ROOT = "Data/mels"

LANGS = ["tamil", "malayalam", "telugu", "kannada"]

SR = 22050
N_FFT = 1024
HOP = 256
N_MELS = 80
# ========================================


def extract_mel(wav_path):
    y, sr = librosa.load(wav_path, sr=SR)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS, fmin=0, fmax=8000
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db


for lang in LANGS:
    print(f"\n🔹 Processing {lang.upper()}")

    wav_dir = os.path.join(DATA_ROOT, lang, "wavs")
    out_dir = os.path.join(OUT_ROOT, lang)

    os.makedirs(out_dir, exist_ok=True)

    wavs = [f for f in os.listdir(wav_dir) if f.endswith(".wav")]

    for wav in tqdm(wavs, desc=f"[{lang.upper()} Mel]"):
        wav_path = os.path.join(wav_dir, wav)
        img_path = os.path.join(out_dir, wav.replace(".wav", ".png"))

        if os.path.exists(img_path):
            continue

        mel = extract_mel(wav_path)

        plt.figure(figsize=(7, 4))
        librosa.display.specshow(
            mel, sr=SR, hop_length=HOP, x_axis="time", y_axis="mel"
        )
        plt.title(wav)
        plt.tight_layout()
        plt.savefig(img_path)
        plt.close()

    print(f"✅ {lang.upper()} mel plots saved to {out_dir}")

print("\n🎉 All multilingual mel-spectrogram plots generated!")
