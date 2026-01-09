import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

plt.rcParams["font.family"] = "Space Mono"

# ================================
BASE = "models"
LANGS = ["tamil", "malayalam", "telugu", "kannada"]

OUT = "models"
os.makedirs(OUT, exist_ok=True)

METRICS = [
    "avg_loss_gen",
    "avg_loss_disc",
    "avg_loss_mel",
    "avg_loss_kl",
    "avg_loss_duration",
]
TITLES = [
    "Generator Loss",
    "Discriminator Loss",
    "Mel-Spectrogram Loss",
    "KL Divergence Loss",
    "Duration Loss",
]

TIME_METRIC = "epoch_time"

# ================================


def latest_run(lang):
    path = os.path.join(BASE, lang)
    runs = sorted([r for r in os.listdir(path) if r.startswith("run-")])
    return os.path.join(path, runs[-1])


def load_metric(run, tag):
    ea = EventAccumulator(run)
    ea.Reload()
    if tag not in ea.Tags()["scalars"]:
        return None
    return [e.value for e in ea.Scalars(tag)]


def align(series):
    min_len = min(len(x) for x in series)
    return [x[:min_len] for x in series]


# ================= PLOT LOSSES =================
for i, metric in enumerate(METRICS):
    train_vals = []
    val_vals = []

    for lang in LANGS:
        run = latest_run(lang)

        train = load_metric(run, f"TrainEpochStats/{metric}")
        val = load_metric(run, f"EvalStats/{metric}")

        if train is None or val is None:
            print(f"⚠ Missing {metric} for {lang}")
            continue

        train_vals.append(train)
        val_vals.append(val)

    train_vals = align(train_vals)
    val_vals = align(val_vals)

    train_mean = np.mean(train_vals, axis=0)
    train_std = np.std(train_vals, axis=0)

    val_mean = np.mean(val_vals, axis=0)
    val_std = np.std(val_vals, axis=0)

    fig = plt.figure()
    ax = fig.gca()

    ax.plot(train_mean, label="Training", linewidth=2)
    ax.fill_between(
        range(len(train_mean)),
        train_mean - train_std,
        train_mean + train_std,
        alpha=0.2,
    )

    ax.plot(val_mean, label="Validation", linewidth=2)
    ax.fill_between(
        range(len(val_mean)), val_mean - val_std, val_mean + val_std, alpha=0.2
    )

    ax.set_title(TITLES[i])
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Values")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/{TITLES[i].replace(' ', '')}.png")
    plt.close()

    print("✅", metric)


# ================= PLOT EPOCH TIME (TRAIN ONLY) =================
time_vals = []

for lang in LANGS:
    run = latest_run(lang)
    t = load_metric(run, f"TrainEpochStats/{TIME_METRIC}")
    if t is not None:
        time_vals.append(t)

time_vals = align(time_vals)
time_mean = np.mean(time_vals, axis=0)
time_std = np.std(time_vals, axis=0)

fig = plt.figure()
ax = fig.gca()
ax.plot(time_mean, label="Training", linewidth=2)
ax.fill_between(
    range(len(time_mean)), time_mean - time_std, time_mean + time_std, alpha=0.2
)
ax.set_xlabel("Epochs")
ax.set_ylabel("Seconds")
ax.set_title("Computation Time")
ax.grid(True)
fig.tight_layout()
fig.savefig(f"{OUT}/Computation.png")
plt.close()

print("✅ epoch_time")

print("\n🎉 Set-1 multilingual training curves ready:", OUT)
