import pandas as pd

LANGS = ["tamil", "malayalam", "telugu", "kannada"]

chars = set()

for lang in LANGS:
    df = pd.read_csv(f"Data/data/{lang}/metadata_phoneme.csv", sep="|", header=None)
    for p in df[2]:
        for ch in p:
            chars.add(ch)

chars = sorted(chars)

with open("chars.txt", "w", encoding="utf-8") as f:
    for c in chars:
        f.write(c + "\n")

print("Total IPA chars:", len(chars))
print("Saved to chars.txt")
