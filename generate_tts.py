import torch
import pandas as pd
import soundfile as sf
import os
from tqdm import tqdm

from TTS.tts.models.vits import Vits
from TTS.tts.configs.vits_config import VitsConfig
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.text.characters import CharactersConfig

LANGS = ["tamil", "malayalam", "telugu", "kannada"]
device = "cuda" if torch.cuda.is_available() else "cpu"

ipa_chars = open("chars.txt", encoding="utf-8").read().splitlines()

for lang in LANGS:
    print("\n🔊 Generating:", lang)

    run = sorted(os.listdir(f"models/{lang}"))[-1]
    run_path = f"models/{lang}/{run}"

    model_path = f"{run_path}/best_model.pth"
    config_path = f"{run_path}/config.json"

    # Load config
    config = VitsConfig()
    config.load_json(config_path)

    # Audio processor
    ap = AudioProcessor.init_from_config(config)

    # Rebuild IPA tokenizer (same as training)
    config.characters = CharactersConfig(
        characters=ipa_chars,
        punctuations="",
        phonemes="",
        pad="<PAD>",
        eos="<EOS>",
        bos="<BOS>",
        blank="<BLANK>",
    )
    config.text_cleaner = None
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # Load model
    model = Vits.init_from_config(config)
    model.load_checkpoint(config, model_path)
    model.to(device)
    model.eval()

    # Load metadata
    df = pd.read_csv(f"Data/data/{lang}/metadata_phoneme.csv", sep="|")

    out_dir = f"Data/generated/{lang}"
    os.makedirs(out_dir, exist_ok=True)

    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"[{lang}] Synthesizing"):
        out_path = f"{out_dir}/{r['wav']}"

        # Skip already generated
        if os.path.exists(out_path):
            continue

        ipa = r["phoneme"]
        ids = tokenizer.text_to_ids(ipa)

        if len(ids) == 0:
            continue

        x = torch.LongTensor(ids)[None].to(device)
        x_len = torch.LongTensor([len(ids)]).to(device)

        with torch.no_grad():
            outputs = model.inference(x, aux_input={"x_lengths": x_len})

        wav = outputs["model_outputs"][0, 0].cpu().numpy().astype("float32")
        wav = wav / max(1.0, abs(wav).max())  # normalize

        sf.write(out_path, wav, ap.sample_rate)

    print(f"✔ {lang} finished")

print("\n🎉 All languages synthesized successfully!")
