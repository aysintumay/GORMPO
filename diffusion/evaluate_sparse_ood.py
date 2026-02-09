#!/usr/bin/env python
"""
Evaluate OOD Detection for Sparse Diffusion Models

This script evaluates OOD detection performance for the 4 sparse diffusion models:
1. halfcheetah-medium-expert-sparse-57.5
2. halfcheetah-medium-expert-sparse-72.5
3. hopper-medium-expert-sparse-78
4. walker2d-medium-expert-sparse-73

Uses the original OOD test data from /public/d4rl/ood_test/{env}-medium-expert-v2/

Results are saved to JSON files for plotting.
"""

import argparse
import json
import os
import pickle
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# Add parent directory for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diffusion_ood import DiffusionOOD
from monte_carlo_sampling_unconditional import build_model_from_ckpt
from ddim_training_unconditional import NPZTargetDataset
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


# Configuration for sparse models
SPARSE_MODELS = {
    "halfcheetah_sparse_57.5": {
        "model_dir": "/public/gormpo/models/halfcheetah_medium_expert_sparse_2/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/halfcheetah-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/halfcheetah_medium_expert_sparse_57.5_train.npz",
        "env_name": "halfcheetah-medium-expert-v2",
        "sparse_level": "57.5%",
    },
    "halfcheetah_sparse_72.5": {
        "model_dir": "/public/gormpo/models/halfcheetah_medium_expert_sparse_3/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/halfcheetah-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/halfcheetah_medium_expert_sparse_72.5_train.npz",
        "env_name": "halfcheetah-medium-expert-v2",
        "sparse_level": "72.5%",
    },
    "hopper_sparse_78": {
        "model_dir": "/public/gormpo/models/hopper_medium_expert_sparse_3/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/hopper-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/hopper_medium_expert_sparse_78_train.npz",
        "env_name": "hopper-medium-expert-v2",
        "sparse_level": "78%",
    },
    "walker2d_sparse_73": {
        "model_dir": "/public/gormpo/models/walker2d_medium_expert_sparse_3/diffusion",
        "ood_base_dir": "/public/d4rl/ood_test/walker2d-medium-expert-v2",
        "train_data": "/public/d4rl/sparse_datasets/diffusion_processed/walker2d_medium_expert_sparse_73_train.npz",
        "env_name": "walker2d-medium-expert-v2",
        "sparse_level": "73%",
    },
}

# OOD distance levels to test (per-environment) — same as run_sparse_ood_eval
OOD_DISTANCES_DEFAULT = [0.5, 1.0, 2.0, 3.0, 4.0]
OOD_DISTANCES_PER_ENV = {
    "halfcheetah": [0.4, 0.6, 0.8, 1.0],
    "hopper": [1.0, 1.2, 1.4, 1.6, 1.8],
    "walker2d": [0.4, 0.6, 0.8, 1.0],
}

# Training config YAMLs to build eval config from (out -> model_dir, npz -> train_data)
TRAINING_CONFIG_MAP = {
    "halfcheetah_sparse_57.5": {
        "yaml": "halfcheetah_mlp_expert_sparse_57.5.yaml",
        "env_name": "halfcheetah-medium-expert-v2",
        "sparse_level": "57.5%",
        "ood_base_dir": "/public/d4rl/ood_test/halfcheetah-medium-expert-v2",
    },
    "hopper_sparse_78": {
        "yaml": "hopper_mlp_expert_sparse_78.yaml",
        "env_name": "hopper-medium-expert-v2",
        "sparse_level": "78%",
        "ood_base_dir": "/public/d4rl/ood_test/hopper-medium-expert-v2",
    },
    "walker2d_sparse_73": {
        "yaml": "walker2d_mlp_expert_sparse_73.yaml",
        "env_name": "walker2d-medium-expert-v2",
        "sparse_level": "73%",
        "ood_base_dir": "/public/d4rl/ood_test/walker2d-medium-expert-v2",
    },
}


def load_models_from_training_configs(config_dir: str) -> Dict[str, Dict]:
    """Load model config from unconditional_training YAMLs (out -> model_dir, npz -> train_data)."""
    try:
        import yaml
    except ImportError:
        raise ImportError("PyYAML required for --from-training-configs. pip install pyyaml")
    models = {}
    for model_name, meta in TRAINING_CONFIG_MAP.items():
        yaml_path = os.path.join(config_dir, meta["yaml"])
        if not os.path.isfile(yaml_path):
            print(f"Warning: {yaml_path} not found, skipping {model_name}")
            continue
        with open(yaml_path, "r") as f:
            cfg = yaml.safe_load(f)
        model_dir = cfg.get("out")
        train_data = cfg.get("npz")
        if not model_dir or not train_data:
            print(f"Warning: {yaml_path} missing 'out' or 'npz', skipping {model_name}")
            continue
        models[model_name] = {
            "model_dir": model_dir,
            "train_data": train_data,
            "ood_base_dir": meta["ood_base_dir"],
            "env_name": meta["env_name"],
            "sparse_level": meta["sparse_level"],
        }
    return models


def load_ood_data(ood_path: str, device: str = "cuda") -> torch.Tensor:
    """Load OOD test data from pickle file."""
    with open(ood_path, "rb") as f:
        data = pickle.load(f)

    # Handle different formats
    if isinstance(data, dict):
        if "target" in data:
            arr = data["target"]
        elif "observations" in data and "actions" in data:
            obs = data["observations"]
            actions = data["actions"]
            arr = np.concatenate([obs, actions], axis=1)
        elif "next_observations" in data and "actions" in data:
            obs = data["next_observations"]
            actions = data["actions"]
            arr = np.concatenate([obs, actions], axis=1)
        else:
            # Try first array-like value
            for v in data.values():
                if isinstance(v, np.ndarray):
                    arr = v
                    break
            else:
                raise ValueError(f"Cannot find data array in {ood_path}")
    elif isinstance(data, np.ndarray):
        arr = data
    else:
        raise ValueError(f"Unknown data format in {ood_path}")

    return torch.FloatTensor(arr).to(device)


def load_train_data(npz_path: str, device: str = "cuda", max_samples: int = 5000) -> torch.Tensor:
    """Load training data from NPZ file."""
    dataset = NPZTargetDataset(npz_path)
    data = dataset.target[:max_samples]
    return data.to(device)


def load_diffusion_model(model_dir: str, device: str = "cuda") -> Tuple[DiffusionOOD, dict]:
    """Load trained diffusion model."""
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    sched_dir = os.path.join(model_dir, "scheduler")

    # Also try model.pt if checkpoint.pt doesn't exist
    if not os.path.exists(ckpt_path):
        alt_ckpt = os.path.join(model_dir, "model.pt")
        if os.path.exists(alt_ckpt):
            ckpt_path = alt_ckpt
        else:
            raise FileNotFoundError(f"No checkpoint found in {model_dir}")

    model, train_cfg = build_model_from_ckpt(ckpt_path, device)

    # Load scheduler
    try:
        scheduler = DDIMScheduler.from_pretrained(sched_dir)
    except Exception:
        try:
            scheduler = DDPMScheduler.from_pretrained(sched_dir)
        except Exception:
            scheduler = DDIMScheduler(
                num_train_timesteps=50000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )

    model.eval()
    ood_model = DiffusionOOD(model, scheduler, device=device, num_inference_steps=100)

    return ood_model, train_cfg


def evaluate_single_ood(
    ood_model: DiffusionOOD,
    train_data: torch.Tensor,
    ood_data: torch.Tensor,
    batch_size: int = 256,
) -> Dict:
    """Evaluate OOD detection for a single OOD dataset (same metrics as run_sparse_ood_eval)."""
    # Score both datasets
    train_scores = ood_model.score_samples(train_data, batch_size=batch_size).cpu().numpy()
    ood_scores = ood_model.score_samples(ood_data, batch_size=batch_size).cpu().numpy()

    # Create labels (0 = in-distribution, 1 = OOD)
    y_true = np.concatenate([
        np.zeros(len(train_scores)),
        np.ones(len(ood_scores))
    ])

    # Use negative log prob as anomaly score (lower log prob = more anomalous)
    y_scores = np.concatenate([-train_scores, -ood_scores])

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    # Compute precision-recall curve and average precision
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    avg_precision = average_precision_score(y_true, y_scores)

    # Accuracy and TPR/TNR at model threshold (same as run_sparse_ood_eval)
    threshold = getattr(ood_model, "threshold", None)
    if threshold is not None:
        # Predict: log_prob < threshold -> OOD (1)
        train_preds = (train_scores < threshold).astype(int)
        ood_preds = (ood_scores < threshold).astype(int)
        y_pred = np.concatenate([train_preds, ood_preds])
        accuracy = float((y_pred == y_true).mean())
        tpr_at_threshold = float(ood_preds.mean())
        tnr_at_threshold = float(1 - train_preds.mean())
    else:
        accuracy = None
        tpr_at_threshold = None
        tnr_at_threshold = None

    # Optimal threshold (Youden's J)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_tpr = float(tpr[optimal_idx])
    optimal_fpr = float(fpr[optimal_idx])
    thresh = thresholds[min(optimal_idx, len(thresholds) - 1)] if len(thresholds) > 0 else -np.inf
    y_pred_optimal = (y_scores >= thresh).astype(int)
    accuracy_optimal = float((y_pred_optimal == y_true).mean())

    results = {
        "roc_auc": float(roc_auc),
        "avg_precision": float(avg_precision),
        "accuracy": accuracy,
        "accuracy_optimal": accuracy_optimal,
        "tpr_at_threshold": tpr_at_threshold,
        "tnr_at_threshold": tnr_at_threshold,
        "optimal_tpr": optimal_tpr,
        "optimal_fpr": optimal_fpr,
        "train_log_prob_mean": float(train_scores.mean()),
        "train_log_prob_std": float(train_scores.std()),
        "ood_log_prob_mean": float(ood_scores.mean()),
        "ood_log_prob_std": float(ood_scores.std()),
        "log_prob_gap": float(train_scores.mean() - ood_scores.mean()),
        "n_train": len(train_scores),
        "n_ood": len(ood_scores),
        "train_scores": train_scores.tolist(),
        "ood_scores": ood_scores.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

    return results


def evaluate_model(
    model_name: str,
    config: Dict,
    ood_distances: List[float],
    device: str = "cuda",
    batch_size: int = 256,
    max_train_samples: int = 5000,
) -> Dict:
    """Evaluate a single sparse diffusion model across all OOD distances."""
    print(f"\n{'='*60}")
    print(f"Evaluating: {model_name}")
    print(f"Model dir: {config['model_dir']}")
    print(f"{'='*60}")

    # Check if model exists
    if not os.path.exists(config["model_dir"]):
        print(f"  [SKIP] Model directory not found: {config['model_dir']}")
        return {"error": "model_not_found", "model_dir": config["model_dir"]}

    # Load model
    try:
        ood_model, _ = load_diffusion_model(config["model_dir"], device)
        print(f"  Loaded diffusion model")
    except Exception as e:
        print(f"  [ERROR] Failed to load model: {e}")
        return {"error": str(e)}

    # Load training data
    try:
        train_data = load_train_data(config["train_data"], device, max_train_samples)
        print(f"  Loaded {len(train_data)} training samples")
    except Exception as e:
        print(f"  [ERROR] Failed to load training data: {e}")
        return {"error": str(e)}

    # Set threshold based on training data
    ood_model.set_threshold(train_data, anomaly_fraction=0.01, batch_size=batch_size)

    results = {
        "model_name": model_name,
        "density_type": "diffusion",
        "env_name": config["env_name"],
        "sparse_level": config["sparse_level"],
        "model_dir": config["model_dir"],
        "train_data_path": config["train_data"],
        "n_train_samples": len(train_data),
        "threshold": ood_model.threshold,
        "ood_distances": ood_distances,
        "ood_results": {},
    }

    # Evaluate each OOD distance
    for dist in ood_distances:
        # Try different filename formats
        ood_files = [
            f"ood-distance-{dist}.pkl",
            f"ood-distance-{int(dist)}.pkl" if dist == int(dist) else None,
            f"ood-distance-{dist}-uniform.pkl",
        ]
        ood_files = [f for f in ood_files if f is not None]

        ood_path = None
        for fname in ood_files:
            candidate = os.path.join(config["ood_base_dir"], fname)
            if os.path.exists(candidate):
                ood_path = candidate
                break

        if ood_path is None:
            print(f"  [SKIP] OOD distance {dist}: file not found")
            results["ood_results"][str(dist)] = {"error": "file_not_found"}
            continue

        try:
            ood_data = load_ood_data(ood_path, device)
            print(f"  Evaluating OOD distance {dist}: {len(ood_data)} samples")

            ood_result = evaluate_single_ood(
                ood_model, train_data, ood_data, batch_size
            )
            ood_result["ood_path"] = ood_path
            results["ood_results"][str(dist)] = ood_result

            print(f"    ROC AUC: {ood_result['roc_auc']:.4f}, "
                  f"Log prob gap: {ood_result['log_prob_gap']:.2f}")

        except Exception as e:
            print(f"  [ERROR] OOD distance {dist}: {e}")
            results["ood_results"][str(dist)] = {"error": str(e)}

    return results


def _get_model_results(model_data: Dict) -> Dict:
    """Get diffusion results from either nested models[name]['diffusion'] or flat models[name]."""
    return model_data.get("diffusion", model_data)


def save_results(results: Dict, output_dir: str, sparse_ood_format: bool = True):
    """Save results to JSON. If sparse_ood_format, also write ood_eval_full_*.json (same as run_sparse_ood_eval)."""
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save full results (same filename as run_sparse_ood_eval when sparse_ood_format)
    if sparse_ood_format:
        full_path = os.path.join(output_dir, f"ood_eval_full_{timestamp}.json")
        latest_name = "ood_eval_latest.json"
    else:
        full_path = os.path.join(output_dir, f"ood_results_full_{timestamp}.json")
        latest_name = "ood_results_full_latest.json"

    with open(full_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull results saved to: {full_path}")

    latest_full = os.path.join(output_dir, latest_name)
    with open(latest_full, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Latest copy saved to: {latest_full}")

    # Summary (without raw scores)
    summary = {}
    for model_name, model_data in results["models"].items():
        model_results = _get_model_results(model_data)
        if "error" in model_results:
            summary[model_name] = model_results
            continue
        summary[model_name] = {
            "env_name": model_results.get("env_name"),
            "sparse_level": model_results.get("sparse_level"),
            "threshold": model_results.get("threshold"),
            "ood_results": {},
        }
        for dist, ood_res in model_results.get("ood_results", {}).items():
            if "error" in ood_res:
                summary[model_name]["ood_results"][dist] = ood_res
            else:
                summary[model_name]["ood_results"][dist] = {
                    "roc_auc": ood_res["roc_auc"],
                    "avg_precision": ood_res["avg_precision"],
                    "train_log_prob_mean": ood_res["train_log_prob_mean"],
                    "ood_log_prob_mean": ood_res["ood_log_prob_mean"],
                    "log_prob_gap": ood_res["log_prob_gap"],
                }

    summary_path = os.path.join(output_dir, f"ood_results_summary_{timestamp}.json")
    with open(summary_path, "w") as f:
        json.dump({"timestamp": timestamp, "models": summary}, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    return full_path, summary_path


def plot_results(results: Dict, output_dir: str):
    """Generate plots from results."""
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: ROC AUC vs OOD distance for all models
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D']

    for idx, (model_name, model_data) in enumerate(results["models"].items()):
        model_results = _get_model_results(model_data)
        if "error" in model_results:
            continue

        distances = []
        aucs = []
        for dist_str, ood_res in model_results.get("ood_results", {}).items():
            if "error" not in ood_res:
                distances.append(float(dist_str))
                aucs.append(ood_res["roc_auc"])

        if distances:
            sorted_pairs = sorted(zip(distances, aucs))
            distances, aucs = zip(*sorted_pairs)

            label = f"{model_results['env_name']} ({model_results['sparse_level']})"
            ax.plot(distances, aucs, marker=markers[idx % len(markers)],
                   color=colors[idx % len(colors)], label=label, linewidth=2, markersize=8)

    ax.set_xlabel("OOD Distance", fontsize=12)
    ax.set_ylabel("ROC AUC", fontsize=12)
    ax.set_title("OOD Detection Performance: Sparse Diffusion Models", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0.5, 1.0])

    plot_path = os.path.join(output_dir, "roc_auc_vs_distance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # Plot 2: Log probability distributions for each model (at distance 1.0)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, (model_name, model_data) in enumerate(results["models"].items()):
        model_results = _get_model_results(model_data)
        if "error" in model_results or idx >= 4:
            continue

        ax = axes[idx]

        # Use distance 1.0 for comparison
        dist_key = "1.0"
        if dist_key not in model_results.get("ood_results", {}):
            dist_key = "1"

        if dist_key in model_results.get("ood_results", {}) and "error" not in model_results["ood_results"][dist_key]:
            ood_res = model_results["ood_results"][dist_key]

            train_scores = np.array(ood_res["train_scores"])
            ood_scores = np.array(ood_res["ood_scores"])

            # Plot histograms
            bins = 50
            ax.hist(train_scores, bins=bins, alpha=0.6, label="In-distribution", color="blue", density=True)
            ax.hist(ood_scores, bins=bins, alpha=0.6, label="OOD (dist=1.0)", color="red", density=True)

            # Add threshold line
            if model_results.get("threshold") is not None:
                ax.axvline(x=model_results["threshold"], color='green', linestyle='--',
                          linewidth=2, label=f'Threshold')

            ax.set_xlabel("Log Probability (ELBO)", fontsize=10)
            ax.set_ylabel("Density", fontsize=10)
            ax.set_title(f"{model_results['env_name']}\n({model_results['sparse_level']} sparse)", fontsize=11)
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "log_prob_distributions.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")

    # Plot 3: ROC curves at distance 1.0
    fig, ax = plt.subplots(figsize=(8, 8))

    for idx, (model_name, model_data) in enumerate(results["models"].items()):
        model_results = _get_model_results(model_data)
        if "error" in model_results:
            continue

        dist_key = "1.0"
        if dist_key not in model_results.get("ood_results", {}):
            dist_key = "1"

        if dist_key in model_results.get("ood_results", {}) and "error" not in model_results["ood_results"][dist_key]:
            ood_res = model_results["ood_results"][dist_key]

            fpr = ood_res["fpr"]
            tpr = ood_res["tpr"]
            roc_auc = ood_res["roc_auc"]

            label = f"{model_results['env_name']} ({model_results['sparse_level']}) - AUC={roc_auc:.3f}"
            ax.plot(fpr, tpr, label=label, linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves at OOD Distance 1.0", fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, alpha=0.3)

    plot_path = os.path.join(output_dir, "roc_curves_dist1.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {plot_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate OOD detection for sparse diffusion models")
    parser.add_argument("--models", nargs="+", default=None,
                       help="Specific models to evaluate (default: all or from --from-training-configs)")
    parser.add_argument("--from-training-configs", action="store_true",
                       help="Load model_dir and train_data from unconditional_training YAMLs (57.5, 78, 73)")
    parser.add_argument("--config-dir", type=str, default=None,
                       help="Directory with unconditional_training YAMLs (default: <repo>/diffusion/configs/unconditional_training)")
    parser.add_argument("--distances", nargs="+", type=float, default=None,
                       help="OOD distances to test (default: per-environment distances)")
    parser.add_argument("--use-env-distances", action="store_true", default=True,
                       help="Use per-environment OOD distances (default: True)")
    parser.add_argument("--output-dir", type=str,
                       default="results/sparse_ood_eval",
                       help="Output directory (default: results/sparse_ood_eval)")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-train-samples", type=int, default=5000)
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")

    args = parser.parse_args()

    # Select models: from training configs or built-in SPARSE_MODELS
    if getattr(args, "from_training_configs", False) or args.config_dir:
        config_dir = args.config_dir
        if not config_dir:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            repo_root = os.path.dirname(script_dir)
            config_dir = os.path.join(repo_root, "diffusion", "configs", "unconditional_training")
        models_to_eval = load_models_from_training_configs(config_dir)
        if args.models:
            models_to_eval = {k: v for k, v in models_to_eval.items() if k in args.models}
    else:
        models_to_eval = SPARSE_MODELS
        if args.models:
            models_to_eval = {k: v for k, v in SPARSE_MODELS.items() if k in args.models}

    # Determine if using per-environment distances
    use_env_distances = args.use_env_distances and args.distances is None

    print("="*60)
    print("Sparse Diffusion Model OOD Evaluation")
    print("="*60)
    print(f"Models: {list(models_to_eval.keys())}")
    if use_env_distances:
        print("OOD distances: per-environment")
        for env, dists in OOD_DISTANCES_PER_ENV.items():
            print(f"  {env}: {dists}")
    else:
        print(f"OOD distances: {args.distances if args.distances else OOD_DISTANCES_DEFAULT}")
    print(f"Device: {args.device}")
    print(f"Output dir: {args.output_dir}")

    # Run evaluation
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "ood_distances": "per-environment" if use_env_distances else (args.distances or OOD_DISTANCES_DEFAULT),
            "ood_distances_per_env": OOD_DISTANCES_PER_ENV if use_env_distances else None,
            "device": args.device,
            "batch_size": args.batch_size,
            "max_train_samples": args.max_train_samples,
        },
        "models": {},
    }

    for model_name, config in models_to_eval.items():
        # Get per-environment distances if enabled
        if use_env_distances:
            env_key = None
            for env in OOD_DISTANCES_PER_ENV.keys():
                if env in model_name.lower():
                    env_key = env
                    break
            distances = OOD_DISTANCES_PER_ENV.get(env_key, OOD_DISTANCES_DEFAULT)
        else:
            distances = args.distances if args.distances else OOD_DISTANCES_DEFAULT

        model_results = evaluate_model(
            model_name, config, distances,
            device=args.device,
            batch_size=args.batch_size,
            max_train_samples=args.max_train_samples,
        )
        all_results["models"][model_name] = {"diffusion": model_results}

    sparse_ood_format = "sparse_ood_eval" in args.output_dir
    full_path, summary_path = save_results(all_results, args.output_dir, sparse_ood_format=sparse_ood_format)

    # Generate plots
    if not args.no_plot:
        plot_results(all_results, args.output_dir)

    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
