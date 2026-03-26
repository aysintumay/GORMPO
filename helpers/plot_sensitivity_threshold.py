"""
Plot threshold sensitivity analysis across all models from saved CSVs.

Each model produces results/abiomed/<algo>_thr_sensitivity/*.csv via
the mult_thr bash scripts. This script reads all CSVs per method, averages
across seeds for each threshold value, and overlays them in one figure with
3 subplots (Return, ACP, Weaning Score) vs threshold_percentile.

Usage:
    python plot_threshold_sensitivity.py [--results_dir results/abiomed] [--out figures/threshold_sensitivity_all_models.png]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

palette = sns.color_palette('colorblind')
colors = [palette[i] for i in [3, 0, 5, 4, 9]]
MODEL_COLORS = {
    "GORMPO-KDE":       colors[0],
    "GORMPO-VAE":       colors[1],
    "GORMPO-RealNVP":   colors[2],
    "GORMPO-NeuralODE": colors[3],
    "GORMPO-Diffusion": colors[4],
}

MODEL_MARKERS = {
    "GORMPO-KDE":       "o",
    "GORMPO-VAE":       "^",
    "GORMPO-RealNVP":   "D",
    "GORMPO-NeuralODE": "v",
    "GORMPO-Diffusion": "P",
}

DIR_TO_MODEL = {
    "mbpo_kde_thr_sensitivity":       "GORMPO-KDE",
    "mbpo_vae_thr_sensitivity":       "GORMPO-VAE",
    "mbpo_realnvp_thr_sensitivity":   "GORMPO-RealNVP",
    "mbpo_neuralode_thr_sensitivity":  "GORMPO-NeuralODE",
    "mbpo_diffusion_thr_sensitivity":  "GORMPO-Diffusion",
}


def plot_metric(ax, x, mean, std, model, ylabel, ylim=None):
    color  = MODEL_COLORS.get(model, None)
    marker = MODEL_MARKERS.get(model, "o")
    ax.plot(x, mean, marker=marker, linewidth=2, label=model, color=color)
    ax.fill_between(x, mean - std, mean + std, alpha=0.2, color=color)
    ax.set_xlabel("Threshold Percentile", fontsize=16)
    ax.set_ylabel(ylabel, fontsize=16)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/abiomed")
    parser.add_argument("--out", default="figures/threshold_sensitivity_all_models.png")
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 1, figsize=(6, 5), dpi=300)
    ax_reward, ax_acp, ax_ws = axes

    found_any = False
    for dir_name, model_name in DIR_TO_MODEL.items():
        dir_path = os.path.join(args.results_dir, dir_name)
        csv_files = sorted(glob.glob(os.path.join(dir_path, "*.csv")))
        if not csv_files:
            print(f"No CSVs found for {model_name} in {dir_path}, skipping.")
            continue

        df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

        # Average across seeds for each threshold
        agg = (
            df.groupby("threshold_percentile")[["mean_return", "mean_acp", "mean_wean_score"]]
            .agg(["mean", "std"])
            .reset_index()
        )
        agg.columns = ["threshold_percentile",
                       "mean_return", "std_return",
                      ]
        agg = agg.sort_values("threshold_percentile")

        x = agg["threshold_percentile"].values
        plot_metric(ax_reward, x, agg["mean_return"].values, agg["std_return"].values, model_name, "Episode Return")
        found_any = True

    if not found_any:
        print(f"No threshold sensitivity CSVs found under {args.results_dir}.")
        return

    for ax, title in zip(axes, ["Return"]):
        ax.set_title(r"$\tau$ Sensitivity for"+title, fontsize=18)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=16,
               bbox_to_anchor=(0.5, -0.08), frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Threshold sensitivity plot saved to {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
