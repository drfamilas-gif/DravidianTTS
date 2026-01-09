import os
import torch
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.models.vits import Vits
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from trainer import Trainer, TrainerArgs
from TTS.tts.utils.text.characters import CharactersConfig


def main():

    LANGS = ["tamil", "malayalam", "telugu", "kannada"]
    BASE_DATA = "Data/data"
    BASE_OUT = "models"

    for lang in LANGS[1:]:
        print("\n====================================")
        print("🚀 Training VITS for", lang.upper())
        print("====================================")

        data_path = f"{BASE_DATA}/{lang}"
        out_path = f"{BASE_OUT}/{lang}"
        os.makedirs(out_path, exist_ok=True)

        dataset_config = BaseDatasetConfig(
            formatter="dravidian",
            meta_file_train="metadata_phoneme.csv",
            meta_file_val="metadata_phoneme.csv",
            path=data_path,
        )

        config = VitsConfig(
            batch_size=16,
            eval_batch_size=8,
            num_loader_workers=2,
            num_eval_loader_workers=2,
            run_eval=True,
            test_delay_epochs=-1,
            epochs=200,
            text_cleaner=None,
            use_phonemes=False,
            print_step=50,
            print_eval=True,
            mixed_precision=True,
            output_path=out_path,
            datasets=[dataset_config],
            save_step=2000,
            phonemizer=None,
        )

        ap = AudioProcessor.init_from_config(config)

        ipa_chars = open("chars.txt", encoding="utf-8").read().splitlines()
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

        train_samples, eval_samples = load_tts_samples(
            dataset_config, eval_split=True, eval_split_size=0.05
        )

        print("Train:", len(train_samples), "Val:", len(eval_samples))

        model = Vits(config, ap, tokenizer, speaker_manager=None)

        trainer = Trainer(
            TrainerArgs(),
            config,
            out_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )

        trainer.fit()


if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
