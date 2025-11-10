import argparse
import os
from typing import Tuple

import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None):
        return iterable

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    import joblib  # type: ignore
except Exception:
    joblib = None


from ddim_training import (
    ConditionalEpsilonMLP,
    ConditionalEpsilonTransformer,
)


def build_model_from_ckpt(ckpt_path: str, device: str) -> Tuple[nn.Module, dict]:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    target_dim = ckpt.get("target_dim")
    cond_dim = ckpt.get("cond_dim")
    model_type = cfg.get("model_type", "mlp")
    time_embed_dim = cfg.get("time_embed_dim", 128)

    if model_type == "mlp":
        model = ConditionalEpsilonMLP(
            target_dim=target_dim,
            cond_dim=cond_dim,
            hidden_dim=cfg.get("hidden_dim", 512),
            time_embed_dim=time_embed_dim,
            num_hidden_layers=cfg.get("num_hidden_layers", 3),
            dropout=cfg.get("dropout", 0.0),
        )
    else:
        model = ConditionalEpsilonTransformer(
            target_dim=target_dim,
            cond_dim=cond_dim,
            d_model=cfg.get("d_model", 256),
            nhead=cfg.get("nhead", 8),
            num_layers=cfg.get("tf_layers", 4),
            dim_feedforward=cfg.get("ff_dim", 512),
            dropout=cfg.get("dropout", 0.1),
            time_embed_dim=time_embed_dim,
        )

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device)
    model.eval()
    return model, cfg


@torch.no_grad()
def ddpm_stochastic_sample(
    model: nn.Module,
    scheduler: DDPMScheduler,
    cond: torch.Tensor,
    target_shape: torch.Size,
    num_inference_steps: int = 1000,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    DDPM stochastic sampling for cond→target with optional generator for reproducibility.
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Generate initial noise with optional generator
    if generator is not None:
        x = torch.randn(target_shape, device=device, generator=generator)
    else:
        x = torch.randn(target_shape, device=device)
    
    for t in scheduler.timesteps:
        t_batch = t.to(device).expand(cond.size(0)).long()
        eps = model(x, cond, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        # DDPM step adds noise at each timestep (stochastic)
        out = scheduler.step(model_output=eps, timestep=t, sample=x, generator=generator)
        x = out.prev_sample
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x


@torch.no_grad()
def ddim_deterministic_sample(
    model: nn.Module,
    scheduler: DDIMScheduler,
    cond: torch.Tensor,
    target_shape: torch.Size,
    num_inference_steps: int = 50,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """
    DDIM deterministic sampling (eta=0) with optional generator for initial noise.
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Generate initial noise with optional generator
    if generator is not None:
        x = torch.randn(target_shape, device=device, generator=generator)
    else:
        x = torch.randn(target_shape, device=device)
    
    for t in scheduler.timesteps:
        t_batch = t.to(device).expand(cond.size(0)).long()
        eps = model(x, cond, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=0.0)
        x = out.prev_sample
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x


@torch.no_grad()
def ddpm_stochastic_sample_batch(
    model: nn.Module,
    scheduler: DDPMScheduler,
    cond: torch.Tensor,
    batch_size: int,
    target_dim: int,
    num_inference_steps: int = 1000,
    device: str = "cpu",
    seed_offset: int = 0,
) -> torch.Tensor:
    """
    Batched DDPM stochastic sampling - processes multiple samples in parallel.
    
    Args:
        model: The noise prediction model
        scheduler: DDPMScheduler instance
        cond: Single condition tensor of shape (1, cond_dim)
        batch_size: Number of samples to generate in parallel
        target_dim: Dimension of target/output
        num_inference_steps: Number of denoising steps
        device: Device to run on
        seed_offset: Offset for random seed generation
        
    Returns:
        Tensor of shape (batch_size, target_dim) with all samples
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Replicate condition for batch
    cond_batch = cond.repeat(batch_size, 1)  # (batch_size, cond_dim)
    
    # Generate initial noise for all samples with different seeds
    # Use different seeds for each sample in the batch
    x = torch.zeros(batch_size, target_dim, device=device)
    for i in range(batch_size):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_offset + i)
        x[i] = torch.randn(target_dim, device=device, generator=generator)
    
    # Get scheduler parameters
    betas = scheduler.betas.to(device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Process all samples through denoising steps
    for step_idx, t in enumerate(scheduler.timesteps):
        t_batch = t.to(device).expand(batch_size).long()
        eps = model(x, cond_batch, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Get timestep index - scheduler timesteps are in descending order (999, 998, ..., 0)
        # Map to actual beta index
        t_val = int(t.item())
        # Find the index in the original timestep array
        # Scheduler uses reversed timesteps, so t_val directly maps to index if timesteps start from num_train_timesteps-1
        # But to be safe, let's use the scheduler's internal step logic
        # Actually, we can use scheduler's scale_model_input to verify, but for DDPM:
        # t_val should be the actual timestep value (0 to num_train_timesteps-1)
        
        # Use scheduler's step_index if available, otherwise compute from timestep
        if hasattr(scheduler, '_index_for_timestep'):
            t_idx = scheduler._index_for_timestep(t, scheduler.timesteps)
        else:
            # Map timestep value to index - assuming timesteps go from num_train_timesteps-1 down to 0
            t_idx = t_val
        
        # Ensure t_idx is within bounds
        t_idx = min(max(0, t_idx), len(betas) - 1)
        
        alpha_t = alphas[t_idx]
        alpha_bar_t = alphas_cumprod[t_idx]
        
        # Compute variance for this step (posterior variance)
        if t_idx > 0:
            alpha_bar_prev = alphas_cumprod[t_idx - 1]
            beta_tilde = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t) * betas[t_idx]
        else:
            beta_tilde = betas[t_idx]
        
        # Predict x_{t-1} mean: 1/sqrt(alpha_t) * (x_t - beta_t/sqrt(1 - alpha_bar_t) * eps)
        pred_prev_sample_mean = (1.0 / torch.sqrt(alpha_t)) * (x - (betas[t_idx] / torch.sqrt(1.0 - alpha_bar_t)) * eps)
        
        # Generate independent noise for each sample in the batch
        # Use manual seed setting for reproducibility while generating in parallel
        noise = torch.zeros_like(x)
        # Generate noise sequentially with different seeds to ensure independence
        # This is fast compared to model forward pass
        for i in range(batch_size):
            generator = torch.Generator(device=device)
            generator.manual_seed(seed_offset + i + step_idx * batch_size * 1000)
            noise[i] = torch.randn(target_dim, device=device, generator=generator)
        
        # Add noise: x_{t-1} = mean + sqrt(beta_tilde) * noise
        x = pred_prev_sample_mean + torch.sqrt(beta_tilde.clamp(min=1e-20)) * noise
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x


@torch.no_grad()
def ddim_deterministic_sample_batch(
    model: nn.Module,
    scheduler: DDIMScheduler,
    cond: torch.Tensor,
    batch_size: int,
    target_dim: int,
    num_inference_steps: int = 50,
    device: str = "cpu",
    seed_offset: int = 0,
) -> torch.Tensor:
    """
    Batched DDIM deterministic sampling - processes multiple samples in parallel.
    Only the initial noise differs; the denoising is deterministic.
    """
    scheduler.set_timesteps(num_inference_steps)
    
    # Replicate condition for batch
    cond_batch = cond.repeat(batch_size, 1)  # (batch_size, cond_dim)
    
    # Generate initial noise for all samples with different seeds
    x = torch.zeros(batch_size, target_dim, device=device)
    for i in range(batch_size):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed_offset + i)
        x[i] = torch.randn(target_dim, device=device, generator=generator)
    
    # Process all samples through denoising steps
    for t in scheduler.timesteps:
        t_batch = t.to(device).expand(batch_size).long()
        eps = model(x, cond_batch, t_batch)
        eps = torch.nan_to_num(eps, nan=0.0, posinf=0.0, neginf=0.0)
        
        # DDIM is deterministic, so we can process all at once
        out = scheduler.step(model_output=eps, timestep=t, sample=x, eta=0.0)
        x = torch.nan_to_num(out.prev_sample, nan=0.0, posinf=0.0, neginf=0.0)
    
    return x


def inverse_scale_target(x_norm: torch.Tensor, mean_arr: np.ndarray | None, std_arr: np.ndarray | None) -> torch.Tensor:
    if mean_arr is None or std_arr is None:
        return x_norm
    mean_t = torch.as_tensor(mean_arr, dtype=x_norm.dtype, device=x_norm.device).view(1, -1)
    std_t = torch.as_tensor(std_arr, dtype=x_norm.dtype, device=x_norm.device).view(1, -1)
    return x_norm * std_t + mean_t


def calculate_nll_binned(
    samples: np.ndarray,
    target: np.ndarray,
    bin_width: float = 0.1,
    smoothing: float = 1e-10,
) -> Tuple[float, np.ndarray, dict]:
    """
    Calculate Negative Log-Likelihood (NLL) using histogram binning.
    
    Args:
        samples: Array of shape (num_samples, target_dim) with Monte Carlo samples
        target: Ground truth target of shape (target_dim,)
        bin_width: Width of each bin (default 0.1)
        smoothing: Small value added to avoid log(0) (default 1e-10)
        
    Returns:
        total_nll: Total NLL summed over all dimensions
        nll_per_dim: NLL for each dimension
        stats: Dictionary with statistics about the binning
    """
    num_samples, target_dim = samples.shape
    target = target.flatten()
    
    # Find the range of values across all samples and target
    all_values = np.concatenate([samples.flatten(), target])
    min_val = np.min(all_values)
    max_val = np.max(all_values)
    
    # Create bins: extend slightly beyond min/max to ensure target is always within range
    margin = bin_width
    bin_edges = np.arange(min_val - margin, max_val + margin + bin_width, bin_width)
    num_bins = len(bin_edges) - 1
    
    nll_per_dim = np.zeros(target_dim)
    bin_probs_per_dim = []
    target_bins_per_dim = []
    
    for dim_idx in range(target_dim):
        # Create histogram for this dimension
        counts, bin_edges_actual = np.histogram(
            samples[:, dim_idx],
            bins=bin_edges,
            density=False
        )
        
        # Convert counts to probability density with smoothing
        # Standard histogram density: density = counts / (total_count * bin_width)
        # With smoothing: density = (counts + smoothing) / (total_smoothed * bin_width)
        # This ensures the integral of density equals 1: sum(density * bin_width) = 1
        smoothed_counts = counts + smoothing
        total_smoothed = smoothed_counts.sum()  # = total_count + smoothing * num_bins
        # Compute density: normalized by total_smoothed and bin_width
        # This removes the influence of bin_width on NLL (NLL = -log(density))
        prob_density = smoothed_counts / (total_smoothed * bin_width)
        # Also compute probability mass for stats (normalized to sum to 1)
        prob_mass = smoothed_counts / total_smoothed
        
        bin_probs_per_dim.append(prob_mass)
        
        # Find which bin the target falls into
        target_val = target[dim_idx]
        # numpy.histogram bins: [edges[i], edges[i+1]) for i < n-1, [edges[n-2], edges[n-1]] for last bin
        # Use side='right' to get the first index where value > target, then subtract 1
        # Special case: if target equals the last edge, it should go in the last bin
        if target_val >= bin_edges_actual[-1]:
            target_bin_idx = len(prob_mass) - 1
        else:
            target_bin_idx = np.searchsorted(bin_edges_actual, target_val, side='right') - 1
            # Clamp to valid range [0, num_bins-1]
            target_bin_idx = max(0, min(target_bin_idx, len(prob_mass) - 1))
        target_bins_per_dim.append(target_bin_idx)
        
        # Calculate NLL from probability density: NLL = -log(pdf)
        # Using density removes the influence of bin_width on NLL values
        # The density is computed as: pdf = (counts + smoothing) / (total * bin_width)
        # So NLL = -log(pdf) is now independent of bin_width (for the same underlying distribution)
        target_density = prob_density[target_bin_idx]
        # Ensure density is positive (smoothing already applied in density calculation)
        nll_per_dim[dim_idx] = -np.log(np.maximum(target_density, 1e-20))
    
    total_nll = nll_per_dim.sum()
    
    stats = {
        'bin_width': bin_width,
        'num_bins': num_bins,
        'bin_edges': bin_edges,
        'bin_probs_per_dim': bin_probs_per_dim,
        'target_bins_per_dim': target_bins_per_dim,
        'value_range': (min_val, max_val),
    }
    
    return total_nll, nll_per_dim, stats


def plot_distributions(
    samples: np.ndarray,
    target: np.ndarray,
    output_path: str,
    num_dims_to_plot: int | None = None,
):
    """
    Plot distribution of Monte Carlo samples.
    
    Args:
        samples: Array of shape (num_samples, target_dim) with all Monte Carlo samples
        target: Ground truth target of shape (target_dim,)
        output_path: Path to save the plot
        num_dims_to_plot: Number of dimensions to plot (None = plot all dimensions)
    """
    num_samples, target_dim = samples.shape
    
    # Limit number of dimensions to plot if specified, otherwise plot all
    if num_dims_to_plot is None:
        dims_to_plot = target_dim
    else:
        dims_to_plot = min(num_dims_to_plot, target_dim)
    
    # Create subplots
    n_cols = 3
    n_rows = (dims_to_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if dims_to_plot > 1 else [axes]
    
    for dim_idx in range(dims_to_plot):
        ax = axes[dim_idx]
        
        # Plot histogram of samples
        ax.hist(samples[:, dim_idx], bins=50, alpha=0.7, density=True, label='Monte Carlo samples')
        
        # Mark ground truth
        ax.axvline(target[dim_idx], color='red', linestyle='--', linewidth=2, label='Ground truth')
        
        # Compute statistics
        mean_sample = samples[:, dim_idx].mean()
        std_sample = samples[:, dim_idx].std()
        
        # Mark mean
        ax.axvline(mean_sample, color='green', linestyle='--', linewidth=2, label=f'Mean: {mean_sample:.4f}')
        
        ax.set_xlabel(f'Dimension {dim_idx}')
        ax.set_ylabel('Density')
        ax.set_title(f'Dim {dim_idx}: μ={mean_sample:.4f}, σ={std_sample:.4f}, GT={target[dim_idx]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(dims_to_plot, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved distribution plot to {output_path}")
    plt.close()
    
    # Also create a summary statistics plot
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Mean and std per dimension
    dims = np.arange(target_dim)
    sample_means = samples.mean(axis=0)
    sample_stds = samples.std(axis=0)
    target_arr = target.flatten()
    
    axes[0].plot(dims, sample_means, 'b-', label='Sample mean', linewidth=2)
    axes[0].plot(dims, target_arr, 'r--', label='Ground truth', linewidth=2)
    axes[0].fill_between(dims, sample_means - sample_stds, sample_means + sample_stds, alpha=0.3, label='±1 std')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Value')
    axes[0].set_title('Mean per Dimension')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Standard deviation per dimension
    axes[1].plot(dims, sample_stds, 'g-', linewidth=2)
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Standard Deviation')
    axes[1].set_title('Std per Dimension')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Error per dimension (mean - ground truth)
    errors = sample_means - target_arr
    axes[2].plot(dims, errors, 'orange', linewidth=2)
    axes[2].axhline(0, color='black', linestyle='--', linewidth=1)
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Error (Mean - GT)')
    axes[2].set_title('Bias per Dimension')
    axes[2].grid(True, alpha=0.3)
    
    summary_path = output_path.replace('.png', '_summary.png')
    plt.tight_layout()
    plt.savefig(summary_path, dpi=150, bbox_inches='tight')
    print(f"Saved summary plot to {summary_path}")
    plt.close()


def main():
    # Stage 1: parse only --config
    config_only = argparse.ArgumentParser(add_help=False)
    config_only.add_argument("--config", type=str, default="", help="YAML with paths and defaults")
    known, _ = config_only.parse_known_args()

    # Load YAML defaults if provided
    yaml_defaults = {}
    if getattr(known, "config", "") and yaml is not None:
        try:
            with open(known.config, "r") as f:
                y = yaml.safe_load(f)
            if isinstance(y, dict):
                yaml_defaults = y
        except Exception:
            pass

    def dget(key, default):
        return yaml_defaults.get(key, default)

    # Stage 2: full parser with YAML-provided defaults (CLI overrides)
    parser = argparse.ArgumentParser(
        description="Monte Carlo sampling for first test case",
        parents=[config_only],
    )
    parser.add_argument("--model-dir", type=str, default=dget("out", ""), help="Dir with checkpoint.pt and scheduler/")
    parser.add_argument("--test-npz", type=str, default=dget("test_npz", ""), help="Path to test NPZ with X_cond/X_target")
    parser.add_argument("--scaler", type=str, default=dget("scaler", ""), help="Joblib StandardScaler for de-normalization")
    parser.add_argument("--num-mc-samples", type=int, default=1000, help="Number of Monte Carlo samples")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for parallel sampling")
    parser.add_argument("--inference-steps", type=int, default=dget("inference_steps", None) or dget("timesteps", 1000))
    parser.add_argument("--scheduler-type", type=str, default="ddpm", choices=["ddpm", "ddim"], help="Scheduler type to use")
    parser.add_argument("--output-dir", type=str, default="./monte_carlo_results", help="Directory to save plots")
    parser.add_argument("--device", type=str, default=dget("device", "cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--num-dims-to-plot", type=int, default=None, help="Number of dimensions to plot histograms for (None = plot all dimensions)")
    parser.add_argument("--bin-width", type=float, default=0.1, help="Bin width for NLL calculation")

    args = parser.parse_args()

    model_dir = args.model_dir
    test_npz = args.test_npz
    scaler_path = args.scaler
    device = args.device
    num_mc_samples = args.num_mc_samples
    scheduler_type = args.scheduler_type.lower()

    if not model_dir:
        raise ValueError("Please provide --model-dir or put 'out' in the YAML config used for training.")
    ckpt_path = os.path.join(model_dir, "checkpoint.pt")
    sched_dir = os.path.join(model_dir, "scheduler")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint at {ckpt_path}")
    if not os.path.exists(sched_dir):
        raise FileNotFoundError(f"Missing scheduler directory at {sched_dir}")
    if not test_npz:
        raise ValueError("Please provide --test-npz or 'test_npz' in YAML config.")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load test data - only first test case
    data = np.load(test_npz)
    cond = data["X_cond"] if "X_cond" in data else data["cond"]
    target = data["X_target"] if "X_target" in data else data["target"]
    
    # Take only the first test case
    cond_first = torch.from_numpy(cond[0:1]).float().view(1, -1).to(device)
    target_first = torch.from_numpy(target[0:1]).float().view(1, -1).to(device)
    target_first_np = target[0:1].flatten()

    print(f"Using first test case:")
    print(f"  Condition shape: {cond_first.shape}")
    print(f"  Target shape: {target_first.shape}")
    print(f"  Number of Monte Carlo samples: {num_mc_samples}")

    # Load scaler for de-normalization
    mean_target_arr = None
    std_target_arr = None
    if scaler_path:
        if joblib is None:
            print("joblib not installed; skipping scaler")
        else:
            try:
                scaler = joblib.load(scaler_path)
                std = np.asarray(getattr(scaler, "scale_", None))
                if std is None:
                    std = np.asarray(getattr(scaler, "std_", None))
                if std is not None:
                    std = np.asarray(std, dtype=np.float64).reshape(-1)
                    std = np.where(std <= 1e-12, 1.0, std)
                    target_dim = target_first.shape[1]
                    std_target = std[-target_dim:]
                    mean = np.asarray(getattr(scaler, "mean_", None))
                    if mean is not None:
                        mean = np.asarray(mean, dtype=np.float64).reshape(-1)
                        mean_target_arr = mean[-target_dim:]
                    std_target_arr = std_target
            except Exception as e:
                print(f"Failed to load scaler: {e}")

    # Build model and scheduler
    model, train_cfg = build_model_from_ckpt(ckpt_path, device)
    
    # Load scheduler
    if scheduler_type == "ddpm":
        try:
            scheduler = DDPMScheduler.from_pretrained(sched_dir)
        except Exception:
            try:
                from diffusers.schedulers.scheduling_ddim import DDIMScheduler
                ddim_scheduler = DDIMScheduler.from_pretrained(sched_dir)
                scheduler = DDPMScheduler(
                    num_train_timesteps=ddim_scheduler.num_train_timesteps,
                    beta_schedule=ddim_scheduler.beta_schedule if hasattr(ddim_scheduler, 'beta_schedule') else "linear",
                    prediction_type=ddim_scheduler.prediction_type,
                )
                print("Converted DDIMScheduler to DDPMScheduler for inference")
            except Exception as e:
                print(f"Warning: Could not load scheduler: {e}")
                print("Creating default DDPMScheduler")
                scheduler = DDPMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="linear",
                    prediction_type="epsilon",
                )
        sampling_fn_batch = ddpm_stochastic_sample_batch
    else:  # ddim
        try:
            from diffusers.schedulers.scheduling_ddim import DDIMScheduler
            scheduler = DDIMScheduler.from_pretrained(sched_dir)
        except Exception:
            print("Warning: Could not load DDIMScheduler, creating default")
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="linear",
                prediction_type="epsilon",
            )
        sampling_fn_batch = ddim_deterministic_sample_batch

    print(f"Using {scheduler_type.upper()} scheduler with {args.inference_steps} inference steps")
    print(f"Batch size: {args.batch_size}")

    # Perform Monte Carlo sampling in batches
    print(f"\nGenerating {num_mc_samples} samples in batches of {args.batch_size}...")
    all_samples = []
    
    target_dim = target_first.shape[1]
    num_batches = (num_mc_samples + args.batch_size - 1) // args.batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Sampling batches"):
        # Calculate actual batch size for last batch
        start_idx = batch_idx * args.batch_size
        end_idx = min(start_idx + args.batch_size, num_mc_samples)
        current_batch_size = end_idx - start_idx
        
        # Generate batch of samples
        batch_samples = sampling_fn_batch(
            model=model,
            scheduler=scheduler,
            cond=cond_first,
            batch_size=current_batch_size,
            target_dim=target_dim,
            num_inference_steps=args.inference_steps,
            device=device,
            seed_offset=start_idx,
        )
        
        # De-normalize if scaler available
        batch_samples_orig = inverse_scale_target(
            batch_samples,
            mean_target_arr,
            std_target_arr
        )
        
        all_samples.append(batch_samples_orig.cpu().numpy())

    all_samples = np.concatenate(all_samples, axis=0)  # Shape: (num_mc_samples, target_dim)
    target_orig = inverse_scale_target(
        target_first,
        mean_target_arr,
        std_target_arr
    ).cpu().numpy().flatten()

    print(f"\nSample statistics:")
    print(f"  Sample shape: {all_samples.shape}")
    print(f"  Mean across samples: {all_samples.mean():.6f}")
    print(f"  Std across samples: {all_samples.std():.6f}")
    print(f"  Ground truth mean: {target_orig.mean():.6f}")

    # Compute per-dimension statistics
    sample_means = all_samples.mean(axis=0)
    sample_stds = all_samples.std(axis=0)
    errors = sample_means - target_orig
    
    print(f"\nPer-dimension statistics:")
    print(f"  Mean error: {errors.mean():.6f} ± {errors.std():.6f}")
    print(f"  Max absolute error: {np.abs(errors).max():.6f}")
    print(f"  Mean std across dims: {sample_stds.mean():.6f}")

    # Calculate NLL using binning
    print(f"\nCalculating NLL with bin width={args.bin_width}...")
    total_nll, nll_per_dim, nll_stats = calculate_nll_binned(
        samples=all_samples,
        target=target_orig,
        bin_width=args.bin_width,
    )
    
    print(f"\nNLL Statistics:")
    print(f"  Total NLL: {total_nll:.6f}")
    print(f"  Mean NLL per dimension: {nll_per_dim.mean():.6f} ± {nll_per_dim.std():.6f}")
    print(f"  Min NLL per dimension: {nll_per_dim.min():.6f}")
    print(f"  Max NLL per dimension: {nll_per_dim.max():.6f}")
    print(f"  Number of bins used: {nll_stats['num_bins']}")
    print(f"  Value range: [{nll_stats['value_range'][0]:.4f}, {nll_stats['value_range'][1]:.4f}]")

    # Save samples to file
    samples_path = os.path.join(args.output_dir, "monte_carlo_samples.npy")
    np.save(samples_path, all_samples)
    print(f"\nSaved samples to {samples_path}")
    
    # Save NLL results
    nll_results_path = os.path.join(args.output_dir, "nll_results.npz")
    np.savez(
        nll_results_path,
        total_nll=total_nll,
        nll_per_dim=nll_per_dim,
        bin_width=args.bin_width,
        num_bins=nll_stats['num_bins'],
        value_range=nll_stats['value_range'],
    )
    print(f"Saved NLL results to {nll_results_path}")

    # Plot distributions
    plot_path = os.path.join(args.output_dir, "monte_carlo_distributions.png")
    plot_distributions(
        samples=all_samples,
        target=target_orig,
        output_path=plot_path,
        num_dims_to_plot=args.num_dims_to_plot,
    )

    # Plot NLL per dimension
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: NLL per dimension
    dims = np.arange(target_dim)
    axes[0].bar(dims, nll_per_dim, alpha=0.7)
    axes[0].axhline(nll_per_dim.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {nll_per_dim.mean():.4f}')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('NLL')
    axes[0].set_title(f'NLL per Dimension (bin width={args.bin_width})')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Histogram of NLL values
    axes[1].hist(nll_per_dim, bins=30, alpha=0.7, edgecolor='black')
    axes[1].axvline(nll_per_dim.mean(), color='red', linestyle='--', linewidth=2, 
                    label=f'Mean: {nll_per_dim.mean():.4f}')
    axes[1].axvline(np.median(nll_per_dim), color='green', linestyle='--', linewidth=2, 
                    label=f'Median: {np.median(nll_per_dim):.4f}')
    axes[1].set_xlabel('NLL')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Distribution of NLL Values')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    nll_plot_path = os.path.join(args.output_dir, "nll_per_dimension.png")
    plt.savefig(nll_plot_path, dpi=150, bbox_inches='tight')
    print(f"Saved NLL plot to {nll_plot_path}")
    plt.close()

    print(f"\nMonte Carlo sampling complete! Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()

