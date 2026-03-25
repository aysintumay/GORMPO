import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

models = ["GORMPO-KDE","GORMPO-VAE","GORMPO-RealNVP","GORMPO-DDPM","GORMPO-NeuralODE"]

palette = sns.color_palette('colorblind')
colors = [palette[i] for i in [3, 0, 5, 4, 9]]

MODEL_COLORS = {
    "GORMPO-KDE": colors[0],
    "GORMPO-VAE": colors[1],
    "GORMPO-RealNVP": colors[2],
    "GORMPO-DDPM": colors[4],
    "GORMPO-NeuralODE": colors[3],
}

# -----------------------
# Penalizer results
# -----------------------
pen_half_mean = [80.41, 89.45, 76.23, 64.27, 82.19]
pen_half_std  = [11.32, 1.91, 6.96, 7.32, 4.52]

pen_hop_mean = [7.77, 6.75, 5.09, 1.53, 1.53]
pen_hop_std  = [4.26, 3.59, 5.83, 0.22, 0.22]

pen_walk_mean = [6.31, 34.73, 9.63, 16.25, 17.10]
pen_walk_std  = [1.26, 23.63, 11.59, 10.45, 24.42]

# -----------------------
# Regularizer results
# -----------------------
reg_half_mean = [79.50, 82.88, 76.89, 66.02, 76.80]
reg_half_std  = [6.95, 5.63, 6.70, 8.12, 4.51]

reg_hop_mean = [12.69, 7.00, 7.63, 4.65, 8.38]
reg_hop_std  = [7.02, 3.30, 7.15, 1.60, 2.91]

reg_walk_mean = [18.94, 13.66, 16.08, 40.48, 2.45]
reg_walk_std  = [22.86, 15.54, 18.02, 32.21, 2.07]

datasets = [
    ("HalfCheetah", pen_half_mean, pen_half_std, reg_half_mean, reg_half_std),
    ("Hopper", pen_hop_mean, pen_hop_std, reg_hop_mean, reg_hop_std),
    ("Walker2d", pen_walk_mean, pen_walk_std, reg_walk_mean, reg_walk_std)
]

x = np.arange(len(models))
width = 0.35

fig, axes = plt.subplots(1, 3, figsize=(18, 7.5), dpi=300)

for ax, (title, pen_mean, pen_std, reg_mean, reg_std) in zip(axes, datasets):
    for i, model in enumerate(models):
        color = MODEL_COLORS[model]

        # Penalizer: solid colored
        ax.bar(
            x[i] - width/2, pen_mean[i], width,
            yerr=pen_std[i], capsize=5,
            color=color, edgecolor="black"
        )

        # Regularizer: hatched, same model color
        ax.bar(
            x[i] + width/2, reg_mean[i], width,
            yerr=reg_std[i], capsize=5,
            facecolor="white", edgecolor=color,
            hatch="//", linewidth=2
        )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=13)
    ax.set_title(title, fontsize=18)
    ax.grid(axis="y", linestyle="--", alpha=0.6)

axes[0].set_ylabel("Return", fontsize=18)

# Legend for model colors
model_handles = [
    Patch(facecolor=MODEL_COLORS[m], edgecolor="black", label=m)
    for m in models
]

# Legend for penalizer vs regularizer
method_handles = [
    Patch(facecolor="black", edgecolor="black", label="Penalizer"),
    Patch(facecolor="white", edgecolor="black", hatch="//", label="Regularizer")
]

legend1 = fig.legend(
    handles=model_handles,
    loc="lower center",
    ncol=5,
    bbox_to_anchor=(0.5, 0.14),
    fontsize=14,
    frameon=True
)

fig.add_artist(legend1)

fig.legend(
    handles=method_handles,
    loc="lower center",
    ncol=2,
    bbox_to_anchor=(0.5, 0.1),
    fontsize=14,
    frameon=True
)

plt.tight_layout(rect=[0, 0.18, 1, 1])
plt.show()