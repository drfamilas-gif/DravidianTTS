import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from PIL import Image
from io import BytesIO

plt.rcParams["font.family"] = "Space Mono"

BASE = "models"
OUT = "models"

LANGS = ["tamil", "malayalam", "telugu", "kannada"]

TAGS = {
    "Alignment": "TrainFigures/alignment",
    "MelSpectrogram": "TrainFigures/trainspectrogram/diff",
    "Speech": "TrainFigures/trainspeech_comparison",
}

os.makedirs(OUT, exist_ok=True)


# ----------------------------
def latest_run(lang):
    p = os.path.join(BASE, lang)
    runs = sorted([r for r in os.listdir(p) if r.startswith("run-")])
    return os.path.join(p, runs[-1])


# ----------------------------
def pick_four(images):
    steps = np.array([x.step for x in images])
    points = np.linspace(steps.min(), steps.max(), 4)

    selected = []
    for p in points:
        idx = np.argmin(np.abs(steps - p))
        selected.append(images[idx])

    return selected, points.astype(int)


# ----------------------------
all_data = {}

for lang in LANGS:
    run = latest_run(lang)
    ea = EventAccumulator(run, size_guidance={"images": 0})
    ea.Reload()

    all_data[lang] = {}
    for key, tag in TAGS.items():
        if tag in ea.Tags()["images"]:
            imgs = ea.Images(tag)
            all_data[lang][key] = pick_four(imgs)

EPOCHS = [0, 75, 125, 200]
TITLES = [
    "Cross-Lingual Attention Alignment Learning in Dravidian TTS",
    "Cross-Lingual MelSpectrogram Learning in Dravidian TTS",
    "Cross-Lingual Speech Learning in Dravidian TTS",
]
# ----------------------------
for i, key in enumerate(TAGS):
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    for r, lang in enumerate(LANGS):
        selected, steps = all_data[lang][key]

        for c, (im_event, step) in enumerate(zip(selected, steps)):
            img = Image.open(BytesIO(im_event.encoded_image_string))
            if key in ["Alignment", "MelSpectrogram"]:
                img = np.asarray(img)[:, :-400, :]
            axes[r, c].imshow(img)
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_visible(False)
            # axes[r, c].axis("off")

            if r == 0:
                axes[r, c].set_title(f"Epoch - {EPOCHS[c]}")
            if c == 0:
                axes[r, c].set_ylabel(LANGS[r].title())

    fig.suptitle(TITLES[i])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{key}.png"))
    plt.close()

    print("✔", key)

print("\n🎉 Multi-language 4×4 panels ready for your paper!")
