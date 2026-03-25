"""
Plot sensitivity analysis across all models from saved CSVs.

Each model produces results/hyperparameter/{algo_name}_sensitivity.csv via tune_gormpo.py.
This script reads all available CSVs and overlays them in one figure
(Return vs reward_penalty_coef) per model.

Usage:
    python plot_sensitivity.py [--results_dir results/hyperparameter] [--out figures/sensitivity_all_models.png]
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# Map CSV model column value -> display name
CSV_TO_MODEL = {
    "gormpo_kde":       "GORMPO-KDE",
    "gormpo_vae":       "GORMPO-VAE",
    "gormpo_realnvp":   "GORMPO-RealNVP",
    "gormpo":           "GORMPO-NeuralODE",
    "gormpo_diffusion": "GORMPO-Diffusion",
}


def plot_metric(ax, df, x_col, mean_col, ci_col, model, ylabel, ylim=None):
    color  = MODEL_COLORS.get(model, None)
    marker = MODEL_MARKERS.get(model, "o")
    x  = df[x_col].values
    y  = df[mean_col].values
    ci = df[ci_col].values
    ax.plot(x, y, marker=marker, linewidth=2, label=model, color=color)
    ax.fill_between(x, y - ci, y + ci, alpha=0.2, color=color)
    ax.set_xlabel("Reward Penalty Coefficient", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10))
    ax.tick_params(axis='both', labelsize=16)
    ax.grid(True, alpha=0.3)
    if ylim is not None:
        ax.set_ylim(*ylim)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/hyperparameter")
    parser.add_argument("--out", default="figures/sensitivity_all_models.png")
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.results_dir, "*_sensitivity.csv")))
    if not csv_files:
        print(f"No sensitivity CSVs found in {args.results_dir}")
        return

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), dpi=300)

    found_any = False
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        if df.empty or "model" not in df.columns:
            continue

        model_key = df["model"].iloc[0]
        model_name = CSV_TO_MODEL.get(model_key, model_key)
        df = df.sort_values("reward_penalty_coef")

        plot_metric(ax, df, "reward_penalty_coef", "mean_reward", "ci95_reward",
                    model_name, "Episode Return")
        found_any = True

    if not found_any:
        print(f"No valid sensitivity CSVs found in {args.results_dir}.")
        return

    ax.set_title(r"$\lambda$ Sensitivity for halfcheetah-medium-expert-sparse", fontsize=18)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3,
               fontsize=16, bbox_to_anchor=(0.5, -0.12), frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    plt.savefig(args.out, dpi=300, bbox_inches="tight")
    print(f"Combined sensitivity plot saved to {args.out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
