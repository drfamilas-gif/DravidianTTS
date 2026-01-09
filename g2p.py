import os
import pandas as pd
from aksharamukha import transliterate
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from tqdm import tqdm

LANGS = {
    "tamil": ("ta", "Tamil"),
    "malayalam": ("ml", "Malayalam"),
    "telugu": ("te", "Telugu"),
    "kannada": ("kn", "Kannada"),
}

BASE = "Data/data"

normalizer = IndicNormalizerFactory()


def indic_to_iso(text, lang_script):
    return transliterate.process(lang_script, "ISO", text)


for lang in LANGS:
    print("\n==============================")
    print("🔹 Processing", lang.upper())
    print("==============================")

    code, script = LANGS[lang]

    meta_path = f"{BASE}/{lang}/metadata.csv"
    out_path = f"{BASE}/{lang}/metadata_phoneme.csv"

    df = pd.read_csv(meta_path, sep="|", names=["wav", "text"])

    norm = normalizer.get_normalizer(code)

    phonemes = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"[{lang.upper()} G2P]"):
        text = str(row["text"])
        text = norm.normalize(text)

        try:
            ph = indic_to_iso(text, script)
        except:
            ph = ""

        phonemes.append(ph)

    df["phoneme"] = phonemes
    df.to_csv(out_path, sep="|", index=False)

    print("✅ Saved:", out_path)

print("\n🎉 All languages converted to phonemes!")
