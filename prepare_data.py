import os
import glob
import shutil
import pandas as pd
import torch
import nemo.collections.asr as nemo_asr
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ================= CONFIG =================
INPUT_ROOT = "Data/source"
OUTPUT_ROOT = "Data/data"

NEMO_MODELS = {
    "tamil": "nemo_weights/indicconformer_stt_ta_hybrid_rnnt_large.nemo",
    "malayalam": "nemo_weights/indicconformer_stt_ml_hybrid_rnnt_large.nemo",
    "telugu": "nemo_weights/indicconformer_stt_te_hybrid_rnnt_large.nemo",
    "kannada": "nemo_weights/indicconformer_stt_kn_hybrid_rnnt_large.nemo",
}

LANG_CODES = {
    "tamil": "ta",
    "malayalam": "ml",
    "telugu": "te",
    "kannada": "kn",
}
# =========================================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔹 Device:", device)

# -------------------------------------------------------
# Loop over languages
# -------------------------------------------------------
for lang, nemo_path in NEMO_MODELS.items():

    print(f"\n==============================")
    print(f"🔹 Processing {lang.upper()}")
    print(f"==============================")

    # Language folders
    lang_root = os.path.join(OUTPUT_ROOT, lang)
    wav_out = os.path.join(lang_root, "wavs")
    meta_csv = os.path.join(lang_root, "metadata.csv")

    os.makedirs(wav_out, exist_ok=True)

    # Load or create metadata
    if os.path.exists(meta_csv):
        df = pd.read_csv(meta_csv, sep="|", header=None, names=["wav", "text"])
        done = set(df["wav"])
        print(f"🔁 Resuming: {len(done)} files already done")
    else:
        df = pd.DataFrame(columns=["wav", "text"])
        done = set()

    # Load NeMo ASR
    print("🔹 Loading NeMo ASR model...")
    model = nemo_asr.models.EncDecRNNTModel.restore_from(nemo_path)
    model.freeze()
    model = model.to(device)
    model.cur_decoder = "ctc"

    # Input wavs
    audio_files = glob.glob(os.path.join(INPUT_ROOT, lang.capitalize(), "*.wav"))
    print(f"🎧 Found {len(audio_files)} wav files")

    for wav in tqdm(audio_files, desc=f"[{lang.upper()} ASR]"):
        base = os.path.basename(wav)

        if base in done:
            continue

        out_wav = os.path.join(wav_out, base)
        shutil.copy(wav, out_wav)

        # ASR
        try:
            text = model.transcribe(
                wav, batch_size=1, logprobs=False, language_id=LANG_CODES[lang]
            )[0][0].strip()
        except Exception as e:
            print("❌ ASR failed:", wav, e)
            continue

        # Append to CSV
        df.loc[len(df)] = [base, text]
        done.add(base)

        # Save every sample (crash-safe)
        df.to_csv(meta_csv, sep="|", index=False, header=False)

    print(f"✅ {lang.upper()} completed: {len(df)} samples")
    print(f"📁 Saved to {lang_root}")

print("\n🎉 All languages processed successfully!")
