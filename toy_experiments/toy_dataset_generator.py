"""
Toy dataset generator using mixture of Gaussians for density model comparison.

This module creates synthetic datasets with controllable difficulty levels
to diagnose and compare loglikelihood scales across different density models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List
import os
from pathlib import Path


class MixtureOfGaussians:
    """Mixture of Gaussians dataset generator for OOD detection experiments."""

    def __init__(self, dim: int = 2, num_components: int = 3, seed: int = 42):
        """
        Initialize mixture of Gaussians.

        Args:
            dim: Dimensionality of the data
            num_components: Number of Gaussian components in the mixture
            seed: Random seed for reproducibility
        """
        self.dim = dim
        self.num_components = num_components
        self.seed = seed
        np.random.seed(seed)

        # Generate random means and covariances for in-distribution
        self.means = []
        self.covariances = []

        for i in range(num_components):
            # Spread means in a circle pattern for 2D, random for higher dims
            if dim == 2:
                angle = 2 * np.pi * i / num_components
                radius = 5.0
                mean = np.array([radius * np.cos(angle), radius * np.sin(angle)])
            else:
                mean = np.random.randn(dim) * 5.0

            # Random covariance matrix (positive definite)
            A = np.random.randn(dim, dim) * 0.5
            cov = A.T @ A + np.eye(dim) * 0.5  # Add diagonal for stability

            self.means.append(mean)
            self.covariances.append(cov)

        # Equal mixing weights
        self.weights = np.ones(num_components) / num_components

    def sample(self, n_samples: int) -> np.ndarray:
        """
        Sample from the mixture of Gaussians.

        Args:
            n_samples: Number of samples to generate

        Returns:
            Array of shape (n_samples, dim)
        """
        # Sample component assignments
        components = np.random.choice(
            self.num_components,
            size=n_samples,
            p=self.weights
        )

        # Sample from each component
        samples = np.zeros((n_samples, self.dim))
        for i in range(self.num_components):
            mask = components == i
            n_comp_samples = mask.sum()
            if n_comp_samples > 0:
                samples[mask] = np.random.multivariate_normal(
                    self.means[i],
                    self.covariances[i],
                    size=n_comp_samples
                )

        return samples

    def log_prob(self, x: np.ndarray) -> np.ndarray:
        """
        Compute true log probability under the mixture.

        Args:
            x: Data points of shape (n_samples, dim)

        Returns:
            Log probabilities of shape (n_samples,)
        """
        n_samples = x.shape[0]
        log_probs = np.zeros(n_samples)

        for i in range(n_samples):
            # Compute probability under each component
            component_probs = []
            for j in range(self.num_components):
                # Multivariate normal PDF
                diff = x[i] - self.means[j]
                cov_inv = np.linalg.inv(self.covariances[j])
                cov_det = np.linalg.det(self.covariances[j])

                exponent = -0.5 * diff @ cov_inv @ diff
                normalization = -0.5 * self.dim * np.log(2 * np.pi) - 0.5 * np.log(cov_det)
                log_prob_component = normalization + exponent

                component_probs.append(np.log(self.weights[j]) + log_prob_component)

            # Log-sum-exp for numerical stability
            log_probs[i] = self._log_sum_exp(component_probs)

        return log_probs

    @staticmethod
    def _log_sum_exp(log_probs: List[float]) -> float:
        """Numerically stable log-sum-exp."""
        max_log_prob = max(log_probs)
        return max_log_prob + np.log(sum(np.exp(lp - max_log_prob) for lp in log_probs))


class OODDatasetGenerator:
    """Generate OOD test sets with different difficulty levels."""

    def __init__(self, in_dist_generator: MixtureOfGaussians):
        """
        Initialize OOD generator.

        Args:
            in_dist_generator: The in-distribution mixture of Gaussians
        """
        self.in_dist = in_dist_generator
        self.dim = in_dist_generator.dim

    def generate_shifted_gaussian(self, n_samples: int, shift_scale: float = 2.0) -> np.ndarray:
        """
        Generate OOD samples by shifting the distribution.

        Args:
            n_samples: Number of OOD samples
            shift_scale: How far to shift (in units of average std dev)

        Returns:
            OOD samples of shape (n_samples, dim)
        """
        # Find the centroid of in-distribution means
        centroid = np.mean(self.in_dist.means, axis=0)

        # Average scale of in-distribution
        avg_scale = np.mean([np.sqrt(np.trace(cov)) for cov in self.in_dist.covariances])

        # Shift away from centroid
        shift = np.random.randn(self.dim)
        shift = shift / np.linalg.norm(shift) * shift_scale * avg_scale * 3

        # Sample from shifted Gaussian
        ood_mean = centroid + shift
        ood_cov = np.eye(self.dim) * (avg_scale ** 2)

        return np.random.multivariate_normal(ood_mean, ood_cov, size=n_samples)

    def generate_uniform_outliers(self, n_samples: int, scale: float = 3.0) -> np.ndarray:
        """
        Generate uniform outliers far from in-distribution.

        Args:
            n_samples: Number of OOD samples
            scale: Scale factor for uniform range

        Returns:
            OOD samples of shape (n_samples, dim)
        """
        # Find range of in-distribution
        all_samples = self.in_dist.sample(10000)
        data_min = all_samples.min(axis=0)
        data_max = all_samples.max(axis=0)
        data_range = data_max - data_min

        # Sample uniformly outside this range
        lower = data_min - scale * data_range
        upper = data_max + scale * data_range

        return np.random.uniform(lower, upper, size=(n_samples, self.dim))

    def generate_sparse_regions(self, n_samples: int) -> np.ndarray:
        """
        Generate OOD samples in sparse regions between modes.

        Args:
            n_samples: Number of OOD samples

        Returns:
            OOD samples of shape (n_samples, dim)
        """
        # Sample pairs of means
        samples = []
        for _ in range(n_samples):
            i, j = np.random.choice(self.in_dist.num_components, size=2, replace=False)
            # Sample at midpoint with small noise
            midpoint = (self.in_dist.means[i] + self.in_dist.means[j]) / 2
            noise = np.random.randn(self.dim) * 0.3
            samples.append(midpoint + noise)

        return np.array(samples)

    def generate_adversarial(self, n_samples: int, epsilon: float = 0.5) -> np.ndarray:
        """
        Generate adversarial samples near the decision boundary.

        Args:
            n_samples: Number of OOD samples
            epsilon: Perturbation magnitude

        Returns:
            OOD samples of shape (n_samples, dim)
        """
        # Start with in-distribution samples
        in_samples = self.in_dist.sample(n_samples)

        # Perturb away from nearest mode
        adversarial = []
        for sample in in_samples:
            # Find nearest mean
            distances = [np.linalg.norm(sample - mean) for mean in self.in_dist.means]
            nearest_idx = np.argmin(distances)

            # Perturb away from nearest mean
            direction = sample - self.in_dist.means[nearest_idx]
            direction = direction / (np.linalg.norm(direction) + 1e-8)

            # Move outward
            adversarial_sample = sample + epsilon * direction * 2
            adversarial.append(adversarial_sample)

        return np.array(adversarial)


def create_toy_datasets(
    dim: int = 2,
    num_components: int = 3,
    n_train: int = 10000,
    n_val: int = 2000,
    n_test: int = 2000,
    n_ood: int = 1000,
    seed: int = 42,
    save_dir: str = "toy_experiments/datasets"
) -> Dict[str, np.ndarray]:
    """
    Create complete toy dataset suite for density model experiments.

    Args:
        dim: Data dimensionality
        num_components: Number of Gaussian components
        n_train: Number of training samples
        n_val: Number of validation samples
        n_test: Number of test samples (in-distribution)
        n_ood: Number of OOD samples per difficulty level
        seed: Random seed
        save_dir: Directory to save datasets

    Returns:
        Dictionary containing all datasets
    """
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # Generate in-distribution data
    mog = MixtureOfGaussians(dim=dim, num_components=num_components, seed=seed)

    train_data = mog.sample(n_train)
    val_data = mog.sample(n_val)
    test_data = mog.sample(n_test)

    # Compute true log probabilities for reference
    train_log_probs = mog.log_prob(train_data)
    val_log_probs = mog.log_prob(val_data)
    test_log_probs = mog.log_prob(test_data)

    # Generate OOD datasets with different difficulty levels
    ood_gen = OODDatasetGenerator(mog)

    # Level 1: Easy - Far outliers (uniform)
    ood_easy = ood_gen.generate_uniform_outliers(n_ood, scale=5.0)

    # Level 2: Medium - Far outliers (uniform)
    ood_medium = ood_gen.generate_uniform_outliers(n_ood, scale=10.0)

    # Level 3: Hard - Sparse regions between modes
    ood_hard = ood_gen.generate_uniform_outliers(n_ood, scale=15.0)

    # Level 4: Very Hard - Adversarial near-boundary samples
    ood_very_hard = ood_gen.generate_uniform_outliers(n_ood, scale=20.0)

    # Compute true log probs for OOD data
    ood_easy_log_probs = mog.log_prob(ood_easy)
    ood_medium_log_probs = mog.log_prob(ood_medium)
    ood_hard_log_probs = mog.log_prob(ood_hard)
    ood_very_hard_log_probs = mog.log_prob(ood_very_hard)

    # Package datasets
    datasets = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'ood_easy': ood_easy,
        'ood_medium': ood_medium,
        'ood_hard': ood_hard,
        'ood_very_hard': ood_very_hard,
        'train_log_probs': train_log_probs,
        'val_log_probs': val_log_probs,
        'test_log_probs': test_log_probs,
        'ood_easy_log_probs': ood_easy_log_probs,
        'ood_medium_log_probs': ood_medium_log_probs,
        'ood_hard_log_probs': ood_hard_log_probs,
        'ood_very_hard_log_probs': ood_very_hard_log_probs,
        'mog_means': np.array(mog.means),
        'mog_covariances': np.array(mog.covariances),
        'mog_weights': mog.weights,
    }

    # Save datasets
    save_path = os.path.join(save_dir, f"toy_mog_dim{dim}_comp{num_components}.npz")
    np.savez(save_path, **datasets)
    print(f"Datasets saved to {save_path}")

    # Print statistics
    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Dimension: {dim}")
    print(f"Number of components: {num_components}")
    print(f"\nSample counts:")
    print(f"  Train: {n_train}")
    print(f"  Validation: {n_val}")
    print(f"  Test (in-dist): {n_test}")
    print(f"  OOD (each level): {n_ood}")

    print(f"\nTrue log-likelihood statistics:")
    print(f"  Train: mean={train_log_probs.mean():.3f}, std={train_log_probs.std():.3f}")
    print(f"  Val: mean={val_log_probs.mean():.3f}, std={val_log_probs.std():.3f}")
    print(f"  Test: mean={test_log_probs.mean():.3f}, std={test_log_probs.std():.3f}")
    print(f"\nOOD true log-likelihoods:")
    print(f"  Easy: mean={ood_easy_log_probs.mean():.3f}, std={ood_easy_log_probs.std():.3f}")
    print(f"  Medium: mean={ood_medium_log_probs.mean():.3f}, std={ood_medium_log_probs.std():.3f}")
    print(f"  Hard: mean={ood_hard_log_probs.mean():.3f}, std={ood_hard_log_probs.std():.3f}")
    print(f"  Very Hard: mean={ood_very_hard_log_probs.mean():.3f}, std={ood_very_hard_log_probs.std():.3f}")
    print("="*60 + "\n")

    return datasets


def visualize_datasets(datasets: Dict[str, np.ndarray], save_path: str = None):
    """
    Visualize 2D toy datasets.

    Args:
        datasets: Dictionary of datasets
        save_path: Path to save figure (optional)
    """
    if datasets['train'].shape[1] != 2:
        print("Visualization only supported for 2D data")
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    # Plot in-distribution data
    axes[0].scatter(datasets['train'][:, 0], datasets['train'][:, 1],
                   alpha=0.3, s=10, c='blue', label='Train')
    axes[0].scatter(datasets['mog_means'][:, 0], datasets['mog_means'][:, 1],
                   c='red', s=200, marker='*', label='True Means', edgecolors='black')
    axes[0].set_title('In-Distribution (Train)', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot test vs each OOD level
    ood_levels = ['easy', 'medium', 'hard', 'very_hard']
    colors = ['green', 'orange', 'purple', 'red']

    for idx, (level, color) in enumerate(zip(ood_levels, colors), start=1):
        ax = axes[idx]

        # In-distribution background
        ax.scatter(datasets['test'][:, 0], datasets['test'][:, 1],
                  alpha=0.2, s=5, c='blue', label='In-Dist')

        # OOD samples
        ood_key = f'ood_{level}'
        ax.scatter(datasets[ood_key][:, 0], datasets[ood_key][:, 1],
                  alpha=0.5, s=20, c=color, label=f'OOD ({level.replace("_", " ").title()})')

        # True means
        ax.scatter(datasets['mog_means'][:, 0], datasets['mog_means'][:, 1],
                  c='red', s=100, marker='*', edgecolors='black', zorder=10)

        ax.set_title(f'OOD Level: {level.replace("_", " ").title()}',
                    fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)

    # Overall comparison
    ax = axes[5]
    ax.scatter(datasets['test'][:, 0], datasets['test'][:, 1],
              alpha=0.1, s=5, c='blue', label='In-Dist')

    for level, color in zip(ood_levels, colors):
        ood_key = f'ood_{level}'
        ax.scatter(datasets[ood_key][:200, 0], datasets[ood_key][:200, 1],
                  alpha=0.4, s=15, c=color, label=level.replace("_", " ").title())

    ax.scatter(datasets['mog_means'][:, 0], datasets['mog_means'][:, 1],
              c='red', s=150, marker='*', edgecolors='black', zorder=10, label='True Means')
    ax.set_title('All OOD Levels Combined', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")

    plt.show()


if __name__ == "__main__":
    # Generate datasets for different dimensionalities
    for dim in [2, 4, 8]:
        print(f"\nGenerating {dim}D datasets...")
        datasets = create_toy_datasets(
            dim=dim,
            num_components=3,
            n_train=10000,
            n_val=2000,
            n_test=2000,
            n_ood=1000,
            seed=42
        )

        # Visualize 2D data
        if dim == 2:
            visualize_datasets(
                datasets,
                save_path=f"toy_experiments/datasets/toy_mog_dim{dim}_visualization.png"
            )
