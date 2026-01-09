import os
import random
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from jiwer import cer
from pesq import pesq
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

SR = 16000
LANGS = ["tamil", "malayalam", "telugu", "kannada"]
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_CSV = f"{OUT_DIR}/metrics.csv"

# ---------------- Whisper ASR ----------------
print("Loading Whisper...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
asr = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").cuda()
asr.eval()


def align(ref, syn):
    T = min(len(ref), len(syn))
    return ref[:T], syn[:T]


def transcribe(wav):
    wav = librosa.resample(wav, orig_sr=22050, target_sr=SR)
    inp = processor(wav, sampling_rate=SR, return_tensors="pt").input_features.cuda()
    with torch.no_grad():
        ids = asr.generate(inp)
    return processor.batch_decode(ids, skip_special_tokens=True)[0]


# ---------------- Metrics ----------------
def mcd(ref, syn):
    ref_mfcc = librosa.feature.mfcc(y=ref, sr=SR, n_mfcc=13)
    syn_mfcc = librosa.feature.mfcc(y=syn, sr=SR, n_mfcc=13)
    T = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
    diff = ref_mfcc[:, :T] - syn_mfcc[:, :T]
    return np.mean(np.sqrt(np.sum(diff**2, axis=0)))


def rmse(ref, syn):
    T = min(len(ref), len(syn))
    return np.sqrt(np.mean((ref[:T] - syn[:T]) ** 2))


def mos_proxy(ref, syn):
    ref = np.log(np.abs(ref) + 1e-6)
    syn = np.log(np.abs(syn) + 1e-6)
    diff = np.mean(np.abs(ref - syn))
    return np.clip(5 - diff * 2, 1, 5)


# ---------------- Main ----------------
rows = []

for lang in LANGS:
    print("\nEvaluating:", lang)

    ref_dir = f"Data/data/{lang}/wavs"
    gen_dir = f"Data/generated/{lang}"

    files = sorted(os.listdir(gen_dir))
    random.shuffle(files)

    split = int(0.7 * len(files))
    train_files = files[:split]
    test_files = files[split:]

    for split_name, file_list in [("train", train_files), ("test", test_files)]:
        mcds, rmses, pesqs, cers, moss = [], [], [], [], []

        for f in tqdm(file_list, desc=f"{lang}-{split_name}"):
            ref, _ = librosa.load(f"{ref_dir}/{f}", sr=SR)
            gen, _ = librosa.load(f"{gen_dir}/{f}", sr=SR)

            ref, gen = align(ref, gen)

            mcds.append(mcd(ref, gen))
            rmses.append(rmse(ref, gen))
            pesqs.append(pesq(SR, ref, gen, "wb"))
            moss.append(mos_proxy(ref, gen))

            ref_txt = transcribe(ref)
            gen_txt = transcribe(gen)
            cers.append(cer(ref_txt, gen_txt))

        rows.append(
            [
                lang,
                split_name,
                np.mean(moss),
                np.mean(mcds),
                np.mean(rmses),
                np.mean(pesqs),
                np.mean(cers),
            ]
        )

# ---------------- Save CSV ----------------
try:
    df = pd.DataFrame(
        rows, columns=["Language", "Split", "MOS", "MCD", "RMSE", "PESQ", "CER"]
    )
    avg = (
        df.groupby("Split")[["MOS", "MCD", "RMSE", "PESQ", "CER"]].mean().reset_index()
    )
    avg["Language"] = "Overall"
    avg = avg[["Language", "Split", "MOS", "MCD", "RMSE", "PESQ", "CER"]]
    df_final = pd.concat([df, avg], ignore_index=True)
    df_final.to_csv(OUT_CSV, index=False)
except:
    df.to_csv(OUT_CSV, index=False)

print("\nSaved:", OUT_CSV)
print(df)
