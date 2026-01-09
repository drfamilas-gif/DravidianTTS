import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

plt.rcParams["font.family"] = "Space Mono"

# ---------------- CONFIG ----------------
CSV = "results/metrics.csv"
OUT = "results"
os.makedirs(OUT, exist_ok=True)

languages = ["Tamil", "Malayalam", "Telugu", "Kannada"]

palette = sns.color_palette("Set2")

# ---------------- Load data ----------------
df = pd.read_csv(CSV)

lang_df = df[~df["Language"].str.contains("Overall", case=False)]
overall_df = df[df["Language"].str.contains("Overall", case=False)]


def get_ylim(values, pad=0.12):
    vmin = min(values)
    vmax = max(values)
    margin = (vmax - vmin) * pad if vmax != vmin else vmax * pad
    return vmin - margin, vmax + margin


# ---------------- Helper for language plots ----------------
def plot_metric(metric, ylabel, filename):
    data = []

    for lang in languages:
        for split in ["Train", "Test"]:
            val = lang_df[(lang_df.Language == lang) & (lang_df.Split == split)][
                metric
            ].values[0]
            data.append([lang, split.capitalize(), val])

    plot_df = pd.DataFrame(data, columns=["Language", "Split", metric])

    fig = plt.figure(figsize=(7, 4.2))
    ax = fig.gca()
    sns.barplot(
        data=plot_df, x="Language", y=metric, hue="Split", palette=palette, ax=ax
    )
    ax.grid(True)
    for p in ax.patches:
        p.set_edgecolor("k")
        p.set_linewidth(1)
        p.set_zorder(1000)
    for c in ax.containers:
        ax.bar_label(c, fmt="%.2f", padding=3)
    ymin, ymax = get_ylim(plot_df[metric].values)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Values")
    ax.set_title(f"{ylabel} across Dravidian Languages")
    fig.tight_layout()
    fig.savefig(f"{OUT}/{filename}")
    plt.close()


# ---------------- Core Evaluation Graphs ----------------
plot_metric("MOS", "Mean Opinion Score", "mos.png")
plot_metric("MCD", "Mel Cepstral Distortion (dB)", "mcd.png")
plot_metric("PESQ", "Perceptual Speech Quality", "pesq.png")
plot_metric("CER", "Character Error Rate", "cer.png")
plot_metric("RMSE", "RMSE", "rmse.png")

# ---------------- Overall Train vs Test ----------------
overall_bar = overall_df.copy()
overall_bar["Split"] = overall_bar["Split"].str.capitalize()

melt_bar = overall_bar.melt(
    id_vars=["Split"],
    value_vars=["MOS", "MCD", "PESQ", "CER", "RMSE"],
    var_name="Metric",
    value_name="Score",
)

fig = plt.figure(figsize=(7, 4.2))
ax = fig.gca()
sns.barplot(data=melt_bar, x="Metric", y="Score", hue="Split", palette=palette, ax=ax)
ax.grid(True)
for p in ax.patches:
    p.set_edgecolor("k")
    p.set_linewidth(1)
    p.set_zorder(1000)
for c in ax.containers:
    ax.bar_label(c, fmt="%.2f", padding=3)
ymin, ymax = get_ylim(melt_bar["Score"].values)
ax.set_ylim(0, ymax)
ax.set_ylabel("Values")
ax.set_title("Overall Performance Comparison")
fig.tight_layout()
fig.savefig(f"{OUT}/overall.png")

print("✅ All Seaborn paper-quality graphs saved in results/")
