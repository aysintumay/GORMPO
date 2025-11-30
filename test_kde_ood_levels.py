"""
Test KDE model on different OOD levels by varying magn parameter.
This script evaluates the model's ability to detect OOD samples at different distances.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from sklearn.metrics import roc_auc_score, accuracy_score

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import KDE model
from kde_module.kde import PercentileThresholdKDE
from common.util import create_synthetic_data


def evaluate_ood_at_magn(model, magn, n_samples=1000, dim=2, anomaly_ratio=0.2, device='cpu'):
    """
    Evaluate KDE model on OOD data generated with a specific magn parameter.

    Args:
        model: Trained KDE model
        magn: Magnitude parameter for OOD data generation
        n_samples: Number of samples to generate
        dim: Dimensionality of data
        anomaly_ratio: Ratio of anomalous samples
        device: Device to use

    Returns:
        Dictionary with evaluation metrics
    """
    # Generate mixed data with specified magn
    mixed_data, labels = create_synthetic_data(
        n_samples=n_samples,
        dim=dim,
        anomaly_type="outlier",
        magn=magn,
        return_mixed=True,
        anomaly_ratio=anomaly_ratio
    )

    # Convert to numpy
    if isinstance(mixed_data, torch.Tensor):
        mixed_data = mixed_data.cpu().numpy()
    labels = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else labels

    # Get log-likelihood scores
    log_probs = model.score_samples(mixed_data)

    # Calculate metrics
    mean_log_likelihood = log_probs.mean()
    std_log_likelihood = log_probs.std()

    # Calculate metrics for normal and anomaly separately
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    normal_log_probs = log_probs[normal_mask]
    anomaly_log_probs = log_probs[anomaly_mask]

    mean_normal = normal_log_probs.mean()
    std_normal = normal_log_probs.std()
    mean_anomaly = anomaly_log_probs.mean()
    std_anomaly = anomaly_log_probs.std()

    # Calculate ROC AUC (higher anomaly score for lower log prob)
    anomaly_scores = -log_probs
    roc_auc = roc_auc_score(labels, anomaly_scores)

    # Calculate accuracy if threshold is available
    if model.threshold is not None:
        predictions = log_probs < model.threshold
        accuracy = accuracy_score(labels, predictions)
    else:
        accuracy = None

    results = {
        'magn': magn,
        'mean_log_likelihood': mean_log_likelihood,
        'std_log_likelihood': std_log_likelihood,
        'mean_normal_log_likelihood': mean_normal,
        'std_normal_log_likelihood': std_normal,
        'mean_anomaly_log_likelihood': mean_anomaly,
        'std_anomaly_log_likelihood': std_anomaly,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'log_probs': log_probs,
        'labels': labels
    }

    return results


def plot_results(all_results, save_dir='figures/kde_ood_magn_tests', model_name='KDE'):
    """
    Plot comprehensive results for different magn values.

    Args:
        all_results: List of result dictionaries
        save_dir: Directory to save plots
        model_name: Name of the model for plot titles
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract data for plotting
    magn_values = [r['magn'] for r in all_results]
    mean_log_likelihoods = [r['mean_log_likelihood'] for r in all_results]
    mean_normal_lls = [r['mean_normal_log_likelihood'] for r in all_results]
    mean_anomaly_lls = [r['mean_anomaly_log_likelihood'] for r in all_results]
    roc_aucs = [r['roc_auc'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results if r['accuracy'] is not None]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Performance on Different OOD Levels (varying magn)',
                 fontsize=16, fontweight='bold')

    # Plot 1: Mean log-likelihood vs magn
    ax1 = axes[0, 0]
    ax1.plot(magn_values, mean_log_likelihoods, 'o-', linewidth=2, markersize=8,
             color='steelblue', label='Overall')
    ax1.plot(magn_values, mean_normal_lls, 's-', linewidth=2, markersize=8,
             color='green', label='Normal samples')
    ax1.plot(magn_values, mean_anomaly_lls, '^-', linewidth=2, markersize=8,
             color='red', label='Anomaly samples')
    ax1.set_xlabel('Distance', fontsize=12)
    ax1.set_ylabel('Mean Log-Likelihood', fontsize=12)
    ax1.set_title('Log-Likelihood vs OOD Distance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ROC AUC vs magn
    ax2 = axes[0, 1]
    ax2.plot(magn_values, roc_aucs, 'o-', linewidth=2, markersize=8, color='forestgreen')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier', alpha=0.7)
    ax2.set_xlabel('Distance', fontsize=12)
    ax2.set_ylabel('ROC AUC', fontsize=12)
    ax2.set_title('ROC AUC vs OOD Distance', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy vs magn (if available)
    ax3 = axes[1, 0]
    if accuracies:
        ax3.plot(magn_values, accuracies, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('Distance', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Classification Accuracy vs OOD Distance', fontsize=13, fontweight='bold')
        ax3.set_ylim([0, 1.05])
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No accuracy data\n(threshold not set)',
                ha='center', va='center', fontsize=14)
        ax3.set_xticks([])
        ax3.set_yticks([])

    # Plot 4: Summary table
    ax4 = axes[1, 1]
    ax4.axis('off')

    table_data = []
    headers = ['Distance', 'Mean LL', 'Normal LL', 'Anomaly LL', 'ROC AUC']

    for r in all_results:
        row = [
            f"{r['magn']:.0f}",
            f"{r['mean_log_likelihood']:.3f}",
            f"{r['mean_normal_log_likelihood']:.3f}",
            f"{r['mean_anomaly_log_likelihood']:.3f}",
            f"{r['roc_auc']:.3f}"
        ]
        table_data.append(row)

    table = ax4.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(headers)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'ood_magn_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot to: {save_path}")
    plt.close()

    # Plot 5: Log-likelihood distributions for each magn
    n_magn = len(all_results)
    n_cols = 3
    n_rows = (n_magn + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'{model_name} Log-Likelihood Distributions for Test Data of Different Gaussian Distances',
                 fontsize=16, fontweight='bold')

    axes = axes.flatten() if n_magn > 1 else [axes]

    for idx, r in enumerate(all_results):
        ax = axes[idx]

        normal_mask = r['labels'] == 0
        anomaly_mask = r['labels'] == 1

        normal_lls = r['log_probs'][normal_mask]
        anomaly_lls = r['log_probs'][anomaly_mask]

        sns.histplot(normal_lls, bins=30, color='green', alpha=0.5,
                    label='Normal', kde=True, ax=ax)
        sns.histplot(anomaly_lls, bins=30, color='red', alpha=0.5,
                    label='Anomaly', kde=True, ax=ax)

        ax.set_xlabel('Log-Likelihood', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distance = {r["magn"]}, ROC AUC = {r["roc_auc"]:.3f}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_magn, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'log_likelihood_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plots to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test KDE on different OOD levels')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved KDE model (without extension)')
    parser.add_argument('--task', type=str, default='hopper-medium-v2',
                       help='Task name for determining input dimensions')
    parser.add_argument('--magn_values', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6],
                       help='List of magn values to test')
    parser.add_argument('--n_samples', type=int, default=1000,
                       help='Number of samples to generate for each test')
    parser.add_argument('--anomaly_ratio', type=float, default=0.2,
                       help='Ratio of anomalous samples in mixed data')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='figures/kde_ood_magn_tests',
                       help='Directory to save results')

    args = parser.parse_args()

    # Set device
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f"CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Using device: {device}")

    # Load model
    print(f"\nLoading KDE model from: {args.model_path}")
    use_gpu = device.startswith('cuda')
    devid = int(device.split(':')[1]) if ':' in device else 0

    model_dict = PercentileThresholdKDE.load_model(args.model_path, use_gpu=use_gpu, devid=devid)
    model = model_dict['model']

    print(f"Model loaded successfully")
    print(f"Model threshold: {model.threshold}")
    print(f"Model training data dimension: {model.training_data.shape[1]}")

    input_dim = model.training_data.shape[1]

    # Test on different magn values
    print("\n" + "="*80)
    print("Testing KDE on Different OOD Levels")
    print("="*80)

    all_results = []

    for magn in args.magn_values:
        print(f"\nTesting magn = {magn}")
        print("-" * 40)

        results = evaluate_ood_at_magn(
            model=model,
            magn=magn,
            n_samples=args.n_samples,
            dim=input_dim,
            anomaly_ratio=args.anomaly_ratio,
            device=device
        )

        all_results.append(results)

        print(f"  Mean log-likelihood: {results['mean_log_likelihood']:.4f}")
        print(f"  Normal samples mean LL: {results['mean_normal_log_likelihood']:.4f}")
        print(f"  Anomaly samples mean LL: {results['mean_anomaly_log_likelihood']:.4f}")
        print(f"  ROC AUC: {results['roc_auc']:.4f}")
        if results['accuracy'] is not None:
            print(f"  Accuracy: {results['accuracy']:.4f}")

    # Create save directory with task name
    save_dir = os.path.join(args.save_dir, args.task.replace('-', '_'))

    # Plot results
    print("\n" + "="*80)
    print("Generating Plots")
    print("="*80)

    plot_results(all_results, save_dir=save_dir, model_name='KDE')

    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    print(f"\nResults saved to: {save_dir}")

    # Print summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Magn':<8} {'Mean LL':<12} {'Normal LL':<12} {'Anomaly LL':<12} {'ROC AUC':<10}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['magn']:<8.0f} {r['mean_log_likelihood']:<12.4f} "
              f"{r['mean_normal_log_likelihood']:<12.4f} "
              f"{r['mean_anomaly_log_likelihood']:<12.4f} {r['roc_auc']:<10.4f}")
    print("-" * 80)


if __name__ == "__main__":
    main()
