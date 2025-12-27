"""
Test Neural ODE model on different OOD levels using pre-generated OOD test datasets.
This script evaluates the model's ability to detect OOD samples at different distances.
The test datasets are loaded from /public/d4rl/ood_test/{dataset_name}/ood-distance-{distance}.pkl
where the first half is ID (in-distribution) and the second half is OOD (out-of-distribution).
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import sys
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pickle
from neuralODE.neural_ode_ood import NeuralODEOOD


def load_ood_test_data(dataset_name, distance, ood_test_dir='/public/d4rl/ood_test'):
    """
    Load OOD test data from pickle file.

    Args:
        dataset_name: Name of the dataset (e.g., 'hopper-medium-v2')
        distance: OOD distance level (e.g., 1, 5, 10, 15)
        ood_test_dir: Directory containing OOD test datasets

    Returns:
        Numpy array of test data where first half is ID and second half is OOD
    """
    # Construct OOD test file path
    # /public/d4rl/ood_test/hopper-medium-v2/ood-distance-5.pkl
    # Handle both int and float distances (5.0 -> 5, 0.5 -> 0.5)
    distance_str = str(int(distance)) if distance == int(distance) else str(distance)
    file_path = os.path.join(ood_test_dir, dataset_name, f'ood-distance-{distance_str}.pkl')

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")

    print(f"Loading test data from: {file_path}")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    # If data is a dictionary, concatenate observations and actions
    if isinstance(data, dict):
        observations = data['observations']
        actions = data['actions']

        # Convert to numpy if needed
        if isinstance(observations, torch.Tensor):
            observations = observations.cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        # Concatenate observations and actions
        data = np.concatenate([observations, actions], axis=1)
    # Convert to numpy if needed
    elif isinstance(data, torch.Tensor):
        data = data.cpu().numpy()

    return data


def evaluate_ood_at_distance(model, dataset_name, distance, ood_test_dir='/public/d4rl/ood_test', device='cpu', mean=None, std=None):
    """
    Evaluate Neural ODE model on OOD test data at a specific distance level.

    Args:
        model: Trained NeuralODEOOD model
        dataset_name: Name of the dataset (e.g., 'hopper-medium-v2')
        distance: OOD distance level
        ood_test_dir: Directory containing OOD test datasets
        device: Device to use
        mean: Mean for normalization (if None, no normalization)
        std: Standard deviation for normalization (if None, no normalization)

    Returns:
        Dictionary with evaluation metrics
    """
    # Load test data
    test_data = load_ood_test_data(dataset_name, distance, ood_test_dir)

    # First half is ID (label 0), second half is OOD (label 1)
    n_samples = len(test_data)
    half_point = n_samples // 2

    # Create labels
    labels = np.zeros(n_samples, dtype=int)
    labels[half_point:] = 1  # Second half is OOD

    print(f"  Total samples: {n_samples} (ID: {half_point}, OOD: {n_samples - half_point})")

    # Normalize if mean and std are provided
    if mean is not None and std is not None:
        test_data = (test_data - mean) / (std + 1e-8)

    # Convert to tensor and move to device
    test_data_tensor = torch.FloatTensor(test_data).to(device)

    # Get log-likelihood scores (batch processing for large datasets)
    model.flow.eval()
    batch_size = 256  # Process in smaller batches to avoid OOM
    all_log_probs = []

    for i in range(0, len(test_data_tensor), batch_size):
        batch = test_data_tensor[i:i+batch_size]
        batch_log_probs = model.score_samples(batch)
        all_log_probs.append(batch_log_probs)

    log_probs = np.concatenate(all_log_probs)

    # Calculate overall metrics
    mean_log_likelihood = log_probs.mean()
    std_log_likelihood = log_probs.std()

    # Calculate metrics for ID and OOD separately
    id_mask = labels == 0
    ood_mask = labels == 1

    id_log_probs = log_probs[id_mask]
    ood_log_probs = log_probs[ood_mask]

    mean_id = id_log_probs.mean()
    std_id = id_log_probs.std()
    mean_ood = ood_log_probs.mean()
    std_ood = ood_log_probs.std()

    # Calculate ROC AUC (higher anomaly score for lower log prob)
    # For OOD detection: ID should have higher log prob (normal), OOD should have lower log prob (anomaly)
    anomaly_scores = -log_probs
    roc_auc = roc_auc_score(labels, anomaly_scores)

    # Calculate ROC curve for plotting
    fpr, tpr, thresholds = roc_curve(labels, anomaly_scores)

    # Calculate accuracy if threshold is available
    if model.threshold is not None:
        # Predict OOD if log_prob < threshold
        predictions = (log_probs < model.threshold).astype(int)
        accuracy = accuracy_score(labels, predictions)
    else:
        accuracy = None

    results = {
        'distance': distance,
        'mean_log_likelihood': mean_log_likelihood,
        'std_log_likelihood': std_log_likelihood,
        'mean_id_log_likelihood': mean_id,
        'std_id_log_likelihood': std_id,
        'mean_ood_log_likelihood': mean_ood,
        'std_ood_log_likelihood': std_ood,
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'log_probs': log_probs,
        'labels': labels,
        'fpr': fpr,
        'tpr': tpr,
        'roc_thresholds': thresholds
    }

    return results


def plot_results(all_results, save_dir='figures/neuralode_ood_distance_tests', model_name='Neural ODE', threshold=None):
    """
    Plot comprehensive results for different OOD distance levels.

    Args:
        all_results: List of result dictionaries
        save_dir: Directory to save plots
        model_name: Name of the model for plot titles
        threshold: Model threshold for OOD detection (optional)
    """
    os.makedirs(save_dir, exist_ok=True)

    # Extract data for plotting
    distance_values = [r['distance'] for r in all_results]
    mean_log_likelihoods = [r['mean_log_likelihood'] for r in all_results]
    mean_id_lls = [r['mean_id_log_likelihood'] for r in all_results]
    mean_ood_lls = [r['mean_ood_log_likelihood'] for r in all_results]
    roc_aucs = [r['roc_auc'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results if r['accuracy'] is not None]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} Performance on Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    # Plot 1: Mean log-likelihood vs distance
    ax1 = axes[0, 0]
    ax1.plot(distance_values, mean_log_likelihoods, 'o-', linewidth=2, markersize=8,
             color='steelblue', label='Overall')
    ax1.plot(distance_values, mean_id_lls, 's-', linewidth=2, markersize=8,
             color='green', label='ID samples')
    ax1.plot(distance_values, mean_ood_lls, '^-', linewidth=2, markersize=8,
             color='red', label='OOD samples')

    # Add threshold line if available
    if threshold is not None:
        ax1.axhline(y=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.3f}')

    ax1.set_xlabel('OOD Distance', fontsize=12)
    ax1.set_ylabel('Mean Log-Likelihood', fontsize=12)
    ax1.set_title('Log-Likelihood vs OOD Distance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: ROC AUC vs distance
    ax2 = axes[0, 1]
    ax2.plot(distance_values, roc_aucs, 'o-', linewidth=2, markersize=8, color='forestgreen')
    ax2.axhline(y=0.5, color='r', linestyle='--', label='Random Classifier', alpha=0.7)
    ax2.set_xlabel('OOD Distance', fontsize=12)
    ax2.set_ylabel('ROC AUC', fontsize=12)
    ax2.set_title('ROC AUC vs OOD Distance', fontsize=13, fontweight='bold')
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Accuracy vs distance (if available)
    ax3 = axes[1, 0]
    if accuracies:
        ax3.plot(distance_values, accuracies, 'o-', linewidth=2, markersize=8, color='purple')
        ax3.set_xlabel('OOD Distance', fontsize=12)
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
    headers = ['Distance', 'Mean LL', 'ID LL', 'OOD LL', 'ROC AUC']

    for r in all_results:
        row = [
            f"{r['distance']:.1f}",
            f"{r['mean_log_likelihood']:.3f}",
            f"{r['mean_id_log_likelihood']:.3f}",
            f"{r['mean_ood_log_likelihood']:.3f}",
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
    save_path = os.path.join(save_dir, 'ood_distance_summary.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot to: {save_path}")
    plt.close()

    # Plot 5: ROC Curves for each distance
    n_distances = len(all_results)
    n_cols = 3
    n_rows = (n_distances + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'{model_name} ROC Curves for Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    if n_distances > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, r in enumerate(all_results):
        ax = axes[idx]

        # Plot ROC curve
        ax.plot(r['fpr'], r['tpr'], linewidth=2, label=f'ROC (AUC = {r["roc_auc"]:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')

        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title(f'Distance = {r["distance"]}', fontsize=12, fontweight='bold')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])

    # Hide unused subplots
    for idx in range(n_distances, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve plots to: {save_path}")
    plt.close()

    # Plot 6: Log-likelihood distributions for each distance
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    fig.suptitle(f'{model_name} Log-Likelihood Distributions for Different OOD Distance Levels',
                 fontsize=16, fontweight='bold')

    if n_distances > 1:
        axes = axes.flatten()
    else:
        axes = [axes]

    for idx, r in enumerate(all_results):
        ax = axes[idx]

        id_mask = r['labels'] == 0
        ood_mask = r['labels'] == 1

        id_lls = r['log_probs'][id_mask]
        ood_lls = r['log_probs'][ood_mask]
        print(f"  Distance {r['distance']}: ID samples = {id_mask.sum()}, OOD samples = {ood_mask.sum()}")
        print(f"  ID log-likelihood range: [{id_lls.min():.3f}, {id_lls.max():.3f}]")
        print(f"  OOD log-likelihood range: [{ood_lls.min():.3f}, {ood_lls.max():.3f}]")
        print(f"  Unique labels: {np.unique(r['labels'])}")
        print(f"  Unique OOD values: {len(np.unique(ood_lls))}, std: {ood_lls.std():.6f}")

        # Determine appropriate bins based on the combined range
        all_lls = np.concatenate([id_lls, ood_lls])
        ll_min, ll_max = all_lls.min(), all_lls.max()

        # Use fixed bins for both distributions
        bins = np.linspace(ll_min, ll_max, 50)

        # Plot ID distribution
        sns.histplot(id_lls, bins=bins, color='green', alpha=0.5,
                    label='ID', kde=True, ax=ax, stat='density')

        # Plot OOD distribution - check if all values are the same
        if ood_lls.std() < 1e-6:
            # All OOD values are essentially the same - plot as a vertical line
            print(f"  WARNING: OOD samples have near-zero variance, plotting as vertical line")
            y_max = ax.get_ylim()[1]
            ax.axvline(x=ood_lls.mean(), color='red', linestyle='-', linewidth=3,
                      alpha=0.7, label=f'OOD (n={len(ood_lls)})')
        else:
            sns.histplot(ood_lls, bins=bins, color='red', alpha=0.5,
                        label='OOD', kde=True, ax=ax, stat='density')

        # Add threshold line if available
        if threshold is not None:
            ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                      label=f'Threshold = {threshold:.3f}')

        ax.set_xlabel('Log-Likelihood', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title(f'Distance = {r["distance"]}, ROC AUC = {r["roc_auc"]:.3f}',
                    fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_distances, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'log_likelihood_distributions.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved distribution plots to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Test Neural ODE on different OOD distance levels')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved Neural ODE model (e.g., neuralODE/hopper_ood)')
    parser.add_argument('--dataset_name', type=str, required=True,
                       help='Dataset name (e.g., hopper-medium-v2)')
    parser.add_argument('--target_dim', type=int, required=True,
                       help='Target dimension (obs_dim + action_dim)')
    parser.add_argument('--distances', type=float, nargs='+', default=[1, 5, 10, 15],
                       help='List of OOD distance values to test')
    parser.add_argument('--ood_test_dir', type=str, default='/public/d4rl/ood_test',
                       help='Directory containing OOD test datasets')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to use (cpu/cuda)')
    parser.add_argument('--save_dir', type=str, default='figures/neuralode_ood_distance_tests',
                       help='Directory to save results')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[512, 512],
                       help='Hidden dimensions of the model')
    parser.add_argument('--activation', type=str, default='silu', choices=['silu', 'tanh'],
                       help='Activation function')
    parser.add_argument('--time_dependent', action='store_true', default=True,
                       help='Whether model is time-dependent')
    parser.add_argument('--solver', type=str, default='dopri5',
                       help='ODE solver')

    args = parser.parse_args()

    # Set device
    device = args.device
    if device.startswith('cuda') and not torch.cuda.is_available():
        print(f"CUDA not available, falling back to CPU")
        device = 'cpu'

    print(f"Using device: {device}")
    print(f"Dataset name: {args.dataset_name}")
    print(f"OOD test directory: {args.ood_test_dir}")

    # Load model
    print(f"\nLoading Neural ODE model from: {args.model_path}")

    # Model path should already be the base path without extension
    # e.g., neuralODE/hopper_ood (not neuralODE/hopper_ood_model.pt)
    model_base_path = args.model_path

    model_dict = NeuralODEOOD.load_model(
        save_path=model_base_path,
        target_dim=args.target_dim,
        hidden_dims=tuple(args.hidden_dims),
        activation=args.activation,
        time_dependent=args.time_dependent,
        solver=args.solver,
        device=device
    )

    model = model_dict['model']
    mean = model_dict.get('mean', None)
    std = model_dict.get('std', None)

    print(f"Model loaded successfully")
    print(f"Model threshold: {model.threshold}")
    print(f"Target dimension: {args.target_dim}")

    # Test on different distance values
    print("\n" + "="*80)
    print("Testing Neural ODE on Different OOD Distance Levels")
    print("="*80)

    all_results = []

    for distance in args.distances:
        print(f"\nTesting distance = {distance}")
        print("-" * 40)

        try:
            results = evaluate_ood_at_distance(
                model=model,
                dataset_name=args.dataset_name,
                distance=distance,
                ood_test_dir=args.ood_test_dir,
                device=device,
                mean=mean,
                std=std
            )

            all_results.append(results)

            print(f"  Mean log-likelihood: {results['mean_log_likelihood']:.4f}")
            print(f"  ID samples mean LL: {results['mean_id_log_likelihood']:.4f}")
            print(f"  OOD samples mean LL: {results['mean_ood_log_likelihood']:.4f}")
            print(f"  ROC AUC: {results['roc_auc']:.4f}")
            if results['accuracy'] is not None:
                print(f"  Accuracy: {results['accuracy']:.4f}")
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            print(f"  Skipping distance {distance}")

    if not all_results:
        print("\nNo test data found! Please check the dataset path and distance values.")
        return

    # Create save directory with dataset name
    # hopper-medium-v2 -> hopper_medium_v2
    dataset_name_normalized = args.dataset_name.replace('-', '_')
    save_dir = os.path.join(args.save_dir, dataset_name_normalized)

    # Plot results
    print("\n" + "="*80)
    print("Generating Plots")
    print("="*80)

    plot_results(all_results, save_dir=save_dir, model_name='Neural ODE', threshold=model.threshold)

    print("\n" + "="*80)
    print("Testing Complete!")
    print("="*80)
    print(f"\nResults saved to: {save_dir}")

    # Print summary
    print("\nSummary:")
    print("-" * 80)
    print(f"{'Distance':<10} {'Mean LL':<12} {'ID LL':<12} {'OOD LL':<12} {'ROC AUC':<10} {'Accuracy':<10}")
    print("-" * 80)
    for r in all_results:
        acc_str = f"{r['accuracy']:.4f}" if r['accuracy'] is not None else "N/A"
        print(f"{r['distance']:<10.1f} {r['mean_log_likelihood']:<12.4f} "
              f"{r['mean_id_log_likelihood']:<12.4f} "
              f"{r['mean_ood_log_likelihood']:<12.4f} {r['roc_auc']:<10.4f} {acc_str:<10}")
    print("-" * 80)


if __name__ == "__main__":
    main()
