"""
Unified training and evaluation script for all density models on toy datasets.

This script trains RealNVP, VAE, Neural ODE, and Diffusion models on the same
toy dataset and compares their loglikelihood scales and OOD detection performance.
"""

import numpy as np
import torch
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import json
from sklearn.metrics import roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import models directly without going through modules that import d4rl
# We'll define lightweight versions here to avoid d4rl dependency issues

# Suppress d4rl warnings
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'

try:
    from realnvp_module.realnvp import RealNVP
except ImportError as e:
    print(f"Warning: Could not import RealNVP: {e}")
    RealNVP = None

try:
    from vae_module.vae import VAE
except ImportError as e:
    print(f"Warning: Could not import VAE: {e}")
    VAE = None

try:
    from neuralODE.neural_ode_density import ContinuousNormalizingFlow, ODEFunc
except ImportError as e:
    print(f"Warning: Could not import Neural ODE: {e}")
    ContinuousNormalizingFlow = None
    ODEFunc = None

try:
    from kde_module.kde import PercentileThresholdKDE
except ImportError as e:
    print(f"Warning: Could not import KDE: {e}")
    PercentileThresholdKDE = None

try:
    from diffusion.ddim_training_unconditional import UnconditionalEpsilonMLP, NPZTargetDataset, log_prob_elbo
    from diffusers.schedulers.scheduling_ddim import DDIMScheduler
except ImportError as e:
    print(f"Warning: Could not import Diffusion modules: {e}")
    UnconditionalEpsilonMLP = None
    DDIMScheduler = None
    log_prob_elbo = None


class DiffusionWrapper:
    """Wrapper for diffusion model to provide score_samples interface."""

    def __init__(self, model, scheduler, device='cuda', num_inference_steps=100):
        self.model = model
        self.scheduler = scheduler
        self.device = device
        self.num_inference_steps = num_inference_steps

    def score_samples(self, x, batch_size=100):
        """Compute log probability using ELBO."""
        if log_prob_elbo is None:
            raise ImportError("log_prob_elbo function not available")

        self.model.eval()
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        x = x.to(self.device)

        # Process in batches
        all_log_probs = []
        for i in range(0, len(x), batch_size):
            batch = x[i:i+batch_size]
            with torch.no_grad():
                log_probs = log_prob_elbo(
                    model=self.model,
                    scheduler=self.scheduler,
                    x0=batch,
                    device=self.device,
                    num_inference_steps=self.num_inference_steps
                )
            all_log_probs.append(log_probs.cpu())

        return torch.cat(all_log_probs).numpy()


class ToyDatasetLoader:
    """Load and prepare toy datasets for training."""

    def __init__(self, dataset_path: str):
        """Load dataset from npz file."""
        self.data = np.load(dataset_path)
        self.dataset_path = dataset_path

    def get_splits(self) -> Dict[str, np.ndarray]:
        """Get train/val/test splits."""
        return {
            'train': self.data['train'],
            'val': self.data['val'],
            'test': self.data['test'],
            'ood_easy': self.data['ood_easy'],
            'ood_medium': self.data['ood_medium'],
            'ood_hard': self.data['ood_hard'],
            'ood_very_hard': self.data['ood_very_hard'],
        }

    def get_true_log_probs(self) -> Dict[str, np.ndarray]:
        """Get ground truth log probabilities."""
        return {
            'train': self.data['train_log_probs'],
            'val': self.data['val_log_probs'],
            'test': self.data['test_log_probs'],
            'ood_easy': self.data['ood_easy_log_probs'],
            'ood_medium': self.data['ood_medium_log_probs'],
            'ood_hard': self.data['ood_hard_log_probs'],
            'ood_very_hard': self.data['ood_very_hard_log_probs'],
        }


class DensityModelTrainer:
    """Train and evaluate all density models."""

    def __init__(
        self,
        dataset_path: str,
        save_dir: str = "toy_experiments/results",
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize trainer.

        Args:
            dataset_path: Path to toy dataset npz file
            save_dir: Directory to save results
            device: Device to use for training
        """
        self.loader = ToyDatasetLoader(dataset_path)
        self.splits = self.loader.get_splits()
        self.true_log_probs = self.loader.get_true_log_probs()
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.input_dim = self.splits['train'].shape[1]
        self.results = {}

        print(f"Initialized trainer with {self.input_dim}D data")
        print(f"Device: {self.device}")
        print(f"Train samples: {len(self.splits['train'])}")
        print(f"Val samples: {len(self.splits['val'])}")
        print(f"Test samples: {len(self.splits['test'])}")

    def train_realnvp(
        self,
        hidden_dims: List[int] = [128, 128],
        num_layers: int = 6,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3
    ) -> RealNVP:
        """Train RealNVP model."""
        print("\n" + "="*60)
        print("TRAINING REALNVP")
        print("="*60)

        model = RealNVP(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            num_layers=num_layers
        )

        # Convert numpy arrays to tensors
        train_tensor = torch.FloatTensor(self.splits['train'])
        val_tensor = torch.FloatTensor(self.splits['val'])

        history = model.fit(
            train_tensor,
            val_tensor,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=20,
            verbose=True
        )

        # Save model
        model_path = self.save_dir / "realnvp_model.pkl"
        model.save_model(str(model_path), train_tensor)
        print(f"Model saved to {model_path}")

        return model

    def train_vae(
        self,
        hidden_dims: List[int] = [128, 128],
        latent_dim: int = 8,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        beta: float = 1.0
    ) -> VAE:
        """Train VAE model."""
        print("\n" + "="*60)
        print("TRAINING VAE")
        print("="*60)

        model = VAE(
            input_dim=self.input_dim,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        )

        # Convert numpy arrays to tensors
        train_tensor = torch.FloatTensor(self.splits['train'])
        val_tensor = torch.FloatTensor(self.splits['val'])
        test_tensor = torch.FloatTensor(self.splits['test'])

        history = model.fit(
            train_tensor,
            val_tensor,
            test_data=test_tensor,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            beta=beta,
            patience=20,
            verbose=True
        )

        # Save model
        model_path = self.save_dir / "vae_model.pkl"
        model.save_model(str(model_path), train_tensor)
        print(f"Model saved to {model_path}")

        return model

    def train_neural_ode(
        self,
        hidden_dims: List[int] = [128, 128],
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
        solver: str = "dopri5",
        time_dependent: bool = True
    ) -> ContinuousNormalizingFlow:
        """Train Neural ODE model."""
        print("\n" + "="*60)
        print("TRAINING NEURAL ODE")
        print("="*60)

        # Create ODE function
        ode_func = ODEFunc(
            dim=self.input_dim,
            hidden_dims=tuple(hidden_dims),
            time_dependent=time_dependent
        ).to(self.device)

        model = ContinuousNormalizingFlow(
            func=ode_func,
            solver=solver
        )

        # Move model to device (including buffers)
        model.to(self.device)

        # Convert data to tensors
        train_tensor = torch.FloatTensor(self.splits['train']).to(self.device)
        val_tensor = torch.FloatTensor(self.splits['val']).to(self.device)

        # Training loop
        optimizer = torch.optim.AdamW(ode_func.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10
        )

        best_val_logprob = -float('inf')
        patience_counter = 0
        max_patience = 20

        for epoch in range(epochs):
            # Training
            ode_func.train()
            train_logprobs = []

            for i in range(0, len(train_tensor), batch_size):
                batch = train_tensor[i:i+batch_size]

                optimizer.zero_grad()
                logprob = model.log_prob(batch)
                loss = -logprob.mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(ode_func.parameters(), 1.0)
                optimizer.step()

                train_logprobs.append(logprob.detach().cpu().numpy())

            # Validation
            ode_func.eval()
            with torch.no_grad():
                val_logprobs = []
                for i in range(0, len(val_tensor), batch_size):
                    batch = val_tensor[i:i+batch_size]
                    logprob = model.log_prob(batch)
                    val_logprobs.append(logprob.detach().cpu().numpy())

            train_logprob_mean = np.concatenate(train_logprobs).mean()
            val_logprob_mean = np.concatenate(val_logprobs).mean()

            scheduler.step(val_logprob_mean)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train LogProb: {train_logprob_mean:.3f}, "
                      f"Val LogProb: {val_logprob_mean:.3f}")

            # Early stopping
            if val_logprob_mean > best_val_logprob:
                best_val_logprob = val_logprob_mean
                patience_counter = 0
                # Save best model
                model_path = self.save_dir / "neural_ode_model.pt"
                torch.save({
                    'ode_func_state_dict': ode_func.state_dict(),
                    'hidden_dims': hidden_dims,
                    'time_dependent': time_dependent,
                    'solver': solver,
                    'input_dim': self.input_dim,
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        checkpoint = torch.load(model_path)
        ode_func.load_state_dict(checkpoint['ode_func_state_dict'])
        print(f"Best model saved to {model_path}")

        return model

    def train_kde(
        self,
        bandwidth: float = 1.0,
        n_neighbors: int = 100,
        percentile: float = 5.0
    ) -> PercentileThresholdKDE:
        """Train KDE model."""
        print("\n" + "="*60)
        print("TRAINING KDE")
        print("="*60)

        model = PercentileThresholdKDE(
            bandwidth=bandwidth,
            n_neighbors=n_neighbors,
            use_gpu=True,
            normalize=True,
            percentile=percentile,
            devid=0 if self.device == "cuda" else -1
        )

        # KDE works with numpy arrays
        model.fit(
            self.splits['train'],
            self.splits['val'],
            verbose=True
        )

        # Save model
        model_path = self.save_dir / "kde_model"
        model.save_model(str(model_path))
        print(f"Model saved to {model_path}")

        return model

    def train_diffusion(
        self,
        hidden_dim: int = 128,
        time_embed_dim: int = 64,
        num_hidden_layers: int = 3,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 2e-4,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 100
    ):
        """
        Train Diffusion model (DDIM with epsilon prediction).

        Args:
            hidden_dim: Hidden dimension for MLP
            time_embed_dim: Time embedding dimension
            num_hidden_layers: Number of hidden layers
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            num_train_timesteps: Number of diffusion timesteps for training
            num_inference_steps: Number of timesteps for ELBO evaluation (faster)

        Returns:
            Tuple of (model, scheduler)
        """
        print("\n" + "="*60)
        print("TRAINING DIFFUSION MODEL (DDIM)")
        print("="*60)

        if UnconditionalEpsilonMLP is None or DDIMScheduler is None:
            print("Diffusion dependencies not available. Skipping...")
            return None, None

        # Create model
        model = UnconditionalEpsilonMLP(
            target_dim=self.input_dim,
            hidden_dim=hidden_dim,
            time_embed_dim=time_embed_dim,
            num_hidden_layers=num_hidden_layers,
            dropout=0.0
        ).to(self.device)

        # Create scheduler
        scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

        # Convert data to tensors
        train_tensor = torch.FloatTensor(self.splits['train']).to(self.device)
        val_tensor = torch.FloatTensor(self.splits['val']).to(self.device)

        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)

        # Move scheduler tensors to device
        scheduler.betas = scheduler.betas.to(self.device)
        scheduler.alphas = scheduler.alphas.to(self.device)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)

        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20

        for epoch in range(epochs):
            # Training
            model.train()
            train_losses = []

            for i in range(0, len(train_tensor), batch_size):
                batch = train_tensor[i:i+batch_size]
                bsz = batch.shape[0]

                # Sample random timesteps
                timesteps = torch.randint(
                    0, num_train_timesteps, (bsz,), device=self.device
                ).long()

                # Sample noise
                noise = torch.randn_like(batch)

                # Get alpha values for these timesteps
                alphas_cumprod = scheduler.alphas_cumprod[timesteps]

                # Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
                sqrt_alpha_prod = alphas_cumprod.sqrt().view(-1, 1)
                sqrt_one_minus_alpha_prod = (1 - alphas_cumprod).sqrt().view(-1, 1)
                noisy_batch = sqrt_alpha_prod * batch + sqrt_one_minus_alpha_prod * noise

                # Predict noise
                optimizer.zero_grad()
                noise_pred = model(noisy_batch, timesteps)

                # MSE loss
                loss = torch.nn.functional.mse_loss(noise_pred, noise)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_losses.append(loss.item())

            # Validation
            model.eval()
            val_losses = []
            with torch.no_grad():
                for i in range(0, len(val_tensor), batch_size):
                    batch = val_tensor[i:i+batch_size]
                    bsz = batch.shape[0]

                    timesteps = torch.randint(
                        0, num_train_timesteps, (bsz,), device=self.device
                    ).long()
                    noise = torch.randn_like(batch)

                    alphas_cumprod = scheduler.alphas_cumprod[timesteps]
                    sqrt_alpha_prod = alphas_cumprod.sqrt().view(-1, 1)
                    sqrt_one_minus_alpha_prod = (1 - alphas_cumprod).sqrt().view(-1, 1)
                    noisy_batch = sqrt_alpha_prod * batch + sqrt_one_minus_alpha_prod * noise

                    noise_pred = model(noisy_batch, timesteps)
                    loss = torch.nn.functional.mse_loss(noise_pred, noise)
                    val_losses.append(loss.item())

            train_loss_mean = np.mean(train_losses)
            val_loss_mean = np.mean(val_losses)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs} - "
                      f"Train Loss: {train_loss_mean:.6f}, "
                      f"Val Loss: {val_loss_mean:.6f}")

            # Early stopping
            if val_loss_mean < best_val_loss:
                best_val_loss = val_loss_mean
                patience_counter = 0
                # Save best model
                model_path = self.save_dir / "diffusion_model.pt"
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'hidden_dim': hidden_dim,
                    'time_embed_dim': time_embed_dim,
                    'num_hidden_layers': num_hidden_layers,
                    'input_dim': self.input_dim,
                    'num_train_timesteps': num_train_timesteps,
                    'num_inference_steps': num_inference_steps,
                }, model_path)
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Load best model
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Best model saved to {model_path}")

        return model, scheduler

    def evaluate_model(
        self,
        model,
        model_name: str,
        use_negative_recon_error: bool = False,
        batch_size: int = 100
    ) -> Dict:
        """
        Evaluate a trained density model on all datasets.

        Args:
            model: Trained density model
            model_name: Name of the model
            use_negative_recon_error: Whether model returns -recon_error (VAE)

        Returns:
            Dictionary of evaluation metrics
        """
        print(f"\n{'='*60}")
        print(f"EVALUATING {model_name.upper()}")
        print(f"{'='*60}")

        results = {
            'model_name': model_name,
            'scores': {},
            'metrics': {}
        }

        # Compute scores for all datasets
        for split_name, data in self.splits.items():
            if model_name == "diffusion" or model_name == "neural_ode":
                # Diffusion and Neural ODE use batching to avoid OOM
                scores = model.score_samples(data, batch_size=batch_size)
            else:
                scores = model.score_samples(data)

            results['scores'][split_name] = scores

            # Get true log probs if available
            if split_name in self.true_log_probs:
                true_scores = self.true_log_probs[split_name]
            else:
                true_scores = None

            # Print statistics
            if scores is not None:
                print(f"\n{split_name}:")
                print(f"  Model scores: mean={scores.mean():.3f}, "
                      f"std={scores.std():.3f}, "
                      f"min={scores.min():.3f}, "
                      f"max={scores.max():.3f}")

                if true_scores is not None:
                    print(f"  True log-prob: mean={true_scores.mean():.3f}, "
                          f"std={true_scores.std():.3f}")

                    # Correlation with true log-probs
                    correlation = np.corrcoef(scores, true_scores)[0, 1]
                    print(f"  Correlation with true log-prob: {correlation:.3f}")
                    results['metrics'][f'{split_name}_correlation'] = correlation

        # Compute OOD detection metrics
        test_scores = results['scores']['test']

        for ood_level in ['easy', 'medium', 'hard', 'very_hard']:
            ood_key = f'ood_{ood_level}'
            ood_scores = results['scores'][ood_key]

            if test_scores is not None and ood_scores is not None:
                # Combine scores
                all_scores = np.concatenate([test_scores, ood_scores])
                labels = np.concatenate([
                    np.ones(len(test_scores)),  # 1 = in-distribution
                    np.zeros(len(ood_scores))   # 0 = out-of-distribution
                ])

                # For negative reconstruction error (VAE), higher is better
                # For log-prob, higher is better
                # So we want to detect OOD as lower scores
                detection_scores = all_scores if not use_negative_recon_error else all_scores

                # Compute metrics
                try:
                    auroc = roc_auc_score(labels, detection_scores)
                    auprc = average_precision_score(labels, detection_scores)

                    results['metrics'][f'{ood_level}_auroc'] = auroc
                    results['metrics'][f'{ood_level}_auprc'] = auprc

                    print(f"\n{ood_level.upper()} OOD Detection:")
                    print(f"  AUROC: {auroc:.4f}")
                    print(f"  AUPRC: {auprc:.4f}")
                except Exception as e:
                    print(f"Error computing metrics for {ood_level}: {e}")

        print(f"{'='*60}\n")
        return results

    def load_realnvp(self) -> RealNVP:
        """Load pre-trained RealNVP model."""
        print("\n" + "="*60)
        print("LOADING REALNVP (PRE-TRAINED)")
        print("="*60)

        model_path = self.save_dir / "realnvp_model.pkl"

        if not (Path(str(model_path) + "_model.pth")).exists():
            print(f"Model not found at {model_path}, training from scratch...")
            return self.train_realnvp()

        # Load metadata to get architecture parameters
        import pickle
        with open(str(model_path) + "_meta_data.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create model with same architecture from metadata
        model = RealNVP(
            input_dim=metadata.get('input_dim', self.input_dim),
            hidden_dims=metadata.get('hidden_dims', [128, 128]),
            num_layers=metadata.get('num_layers', 6)
        )

        # Load weights and move to device
        model.load_state_dict(torch.load(str(model_path) + "_model.pth", map_location=self.device))
        model.to(self.device)
        model.eval()

        print(f"Model loaded from {model_path} on device {self.device}")
        print(f"  Input dim: {metadata.get('input_dim')}, Num layers: {metadata.get('num_layers')}")
        return model

    def load_vae(self) -> VAE:
        """Load pre-trained VAE model."""
        print("\n" + "="*60)
        print("LOADING VAE (PRE-TRAINED)")
        print("="*60)

        model_path = self.save_dir / "vae_model.pkl"

        if not (Path(str(model_path) + "_model.pth")).exists():
            print(f"Model not found at {model_path}, training from scratch...")
            return self.train_vae()

        # Load metadata to get architecture parameters
        import pickle
        with open(str(model_path) + "_meta_data.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create model with same architecture from metadata
        model = VAE(
            input_dim=metadata.get('input_dim', self.input_dim),
            hidden_dims=metadata.get('hidden_dims', [128, 128]),
            latent_dim=metadata.get('latent_dim', max(2, self.input_dim // 2))
        )

        # Load weights and move to device
        model.load_state_dict(torch.load(str(model_path) + "_model.pth", map_location=self.device))
        model.to(self.device)
        model.eval()

        print(f"Model loaded from {model_path} on device {self.device}")
        print(f"  Input dim: {metadata.get('input_dim')}, Latent dim: {metadata.get('latent_dim')}")
        return model

    def load_kde(self) -> PercentileThresholdKDE:
        """Load pre-trained KDE model."""
        print("\n" + "="*60)
        print("LOADING KDE (PRE-TRAINED)")
        print("="*60)

        model_path = self.save_dir / "kde_model.pkl"

        if not model_path.exists():
            print(f"Model not found at {model_path}, training from scratch...")
            return self.train_kde()

        # Load KDE model
        model = PercentileThresholdKDE.load_model(str(self.save_dir / "kde_model"))

        print(f"Model loaded from {model_path}")
        return model

    def load_neural_ode(self) -> ContinuousNormalizingFlow:
        """Load pre-trained Neural ODE model."""
        print("\n" + "="*60)
        print("LOADING NEURAL ODE (PRE-TRAINED)")
        print("="*60)

        model_path = self.save_dir / "neural_ode_model.pt"

        if not model_path.exists():
            print(f"Model not found at {model_path}, training from scratch...")
            return self.train_neural_ode()

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Create ODE function
        ode_func = ODEFunc(
            dim=checkpoint['input_dim'],
            hidden_dims=tuple(checkpoint['hidden_dims']),
            time_dependent=checkpoint['time_dependent']
        ).to(self.device)

        ode_func.load_state_dict(checkpoint['ode_func_state_dict'])

        model = ContinuousNormalizingFlow(
            func=ode_func,
            solver=checkpoint['solver']
        )

        # Move entire model to device (including buffers like integration_times)
        model.to(self.device)

        print(f"Model loaded from {model_path} on device {self.device}")
        return model

    def load_diffusion(self):
        """Load pre-trained Diffusion model."""
        print("\n" + "="*60)
        print("LOADING DIFFUSION (PRE-TRAINED)")
        print("="*60)

        model_path = self.save_dir / "diffusion_model.pt"

        if not model_path.exists():
            print(f"Model not found at {model_path}, training from scratch...")
            return self.train_diffusion()

        if UnconditionalEpsilonMLP is None or DDIMScheduler is None:
            print("Diffusion dependencies not available. Skipping...")
            return None

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')

        # Create model
        model = UnconditionalEpsilonMLP(
            target_dim=checkpoint['input_dim'],
            hidden_dim=checkpoint['hidden_dim'],
            time_embed_dim=checkpoint['time_embed_dim'],
            num_hidden_layers=checkpoint['num_hidden_layers'],
            dropout=0.0
        ).to(self.device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Create scheduler
        scheduler = DDIMScheduler(
            num_train_timesteps=checkpoint['num_train_timesteps'],
            beta_schedule="linear",
            prediction_type="epsilon"
        )

        # Move scheduler tensors to device
        scheduler.betas = scheduler.betas.to(self.device)
        scheduler.alphas = scheduler.alphas.to(self.device)
        scheduler.alphas_cumprod = scheduler.alphas_cumprod.to(self.device)

        print(f"Model loaded from {model_path} on device {self.device}")
        print(f"  Input dim: {checkpoint['input_dim']}, Inference steps: {checkpoint['num_inference_steps']}")

        return DiffusionWrapper(
            model=model,
            scheduler=scheduler,
            device=self.device,
            num_inference_steps=checkpoint['num_inference_steps']
        )

    def run_all_experiments(self):
        """Train and evaluate all models."""
        print("\n" + "="*80)
        print("STARTING COMPREHENSIVE DENSITY MODEL COMPARISON")
        print("="*80)

        # Load or train RealNVP
        realnvp_model = self.load_realnvp()
        self.results['realnvp'] = self.evaluate_model(realnvp_model, "realnvp")

        # Load or train VAE
        vae_model = self.load_vae()
        self.results['vae'] = self.evaluate_model(
            vae_model,
            "vae",
            use_negative_recon_error=True
        )

        # Load or train KDE
        kde_model = self.load_kde()
        self.results['kde'] = self.evaluate_model(kde_model, "kde")

        # Load or train Neural ODE
        neural_ode_model = self.load_neural_ode()
        # Use smaller batch size for Neural ODE to avoid OOM during ODE integration
        self.results['neural_ode'] = self.evaluate_model(neural_ode_model, "neural_ode", batch_size=50)

        # Load or train Diffusion
        diffusion_model = self.load_diffusion()
        if diffusion_model is not None:
            # Use smaller batch size for Diffusion to avoid OOM during ELBO computation
            self.results['diffusion'] = self.evaluate_model(diffusion_model, "diffusion", batch_size=50)

        # Save results
        self.save_results()
        self.create_visualizations()

    def save_results(self):
        """Save results to JSON."""
        # Convert numpy arrays to lists for JSON serialization
        results_serializable = {}

        for model_name, model_results in self.results.items():
            results_serializable[model_name] = {
                'model_name': model_results['model_name'],
                'metrics': model_results['metrics'],
                'score_statistics': {}
            }

            # Save score statistics instead of full arrays
            for split_name, scores in model_results['scores'].items():
                if scores is not None:
                    results_serializable[model_name]['score_statistics'][split_name] = {
                        'mean': float(scores.mean()),
                        'std': float(scores.std()),
                        'min': float(scores.min()),
                        'max': float(scores.max()),
                        'median': float(np.median(scores)),
                        'q25': float(np.percentile(scores, 25)),
                        'q75': float(np.percentile(scores, 75)),
                    }

        save_path = self.save_dir / "results_summary.json"
        with open(save_path, 'w') as f:
            json.dump(results_serializable, f, indent=2)

        print(f"\nResults saved to {save_path}")

    def create_visualizations(self):
        """Create comparison visualizations."""
        print("\nCreating visualizations...")

        # 1. Loglikelihood comparison across models
        self._plot_loglikelihood_comparison()

        # 2. OOD detection performance
        self._plot_ood_detection_performance()

        # 3. Distribution of scores
        self._plot_score_distributions()

        print("Visualizations complete!")

    def _plot_loglikelihood_comparison(self):
        """Plot loglikelihood scores across models and datasets."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        splits_to_plot = ['test', 'ood_easy', 'ood_medium', 'ood_hard']
        model_names = list(self.results.keys())

        for idx, split_name in enumerate(splits_to_plot):
            ax = axes[idx]

            # Collect data for boxplot
            data_to_plot = []
            labels = []

            for model_name in model_names:
                scores = self.results[model_name]['scores'].get(split_name)
                if scores is not None:
                    data_to_plot.append(scores)
                    labels.append(model_name)

            # Add true log-probs
            if split_name in self.true_log_probs:
                data_to_plot.append(self.true_log_probs[split_name])
                labels.append('true')

            # Create boxplot
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)

            # Color boxes
            colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink', 'lightsalmon']
            for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
                patch.set_facecolor(color)

            ax.set_title(f'{split_name.replace("_", " ").title()}',
                        fontsize=14, fontweight='bold')
            ax.set_ylabel('Log-Likelihood / Score', fontsize=12)
            ax.grid(alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        save_path = self.save_dir / "loglikelihood_comparison.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def _plot_ood_detection_performance(self):
        """Plot OOD detection performance metrics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        ood_levels = ['easy', 'medium', 'hard', 'very_hard']
        model_names = list(self.results.keys())
        x = np.arange(len(ood_levels))
        # Dynamically calculate bar width based on number of models
        width = 0.8 / len(model_names) if len(model_names) > 0 else 0.2

        # AUROC
        for idx, model_name in enumerate(model_names):
            aurocs = [
                self.results[model_name]['metrics'].get(f'{level}_auroc', 0)
                for level in ood_levels
            ]
            ax1.bar(x + idx * width, aurocs, width, label=model_name)

        ax1.set_xlabel('OOD Difficulty Level', fontsize=12)
        ax1.set_ylabel('AUROC', fontsize=12)
        ax1.set_title('OOD Detection: AUROC', fontsize=14, fontweight='bold')
        ax1.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax1.set_xticklabels([l.replace('_', ' ').title() for l in ood_levels])
        ax1.legend()
        ax1.grid(alpha=0.3, axis='y')
        ax1.set_ylim([0, 1])

        # AUPRC
        for idx, model_name in enumerate(model_names):
            auprcs = [
                self.results[model_name]['metrics'].get(f'{level}_auprc', 0)
                for level in ood_levels
            ]
            ax2.bar(x + idx * width, auprcs, width, label=model_name)

        ax2.set_xlabel('OOD Difficulty Level', fontsize=12)
        ax2.set_ylabel('AUPRC', fontsize=12)
        ax2.set_title('OOD Detection: AUPRC', fontsize=14, fontweight='bold')
        ax2.set_xticks(x + width * (len(model_names) - 1) / 2)
        ax2.set_xticklabels([l.replace('_', ' ').title() for l in ood_levels])
        ax2.legend()
        ax2.grid(alpha=0.3, axis='y')
        ax2.set_ylim([0, 1])

        plt.tight_layout()
        save_path = self.save_dir / "ood_detection_performance.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()

    def _plot_score_distributions(self):
        """Plot score distributions for each model."""
        model_names = list(self.results.keys())
        n_models = len(model_names)

        fig, axes = plt.subplots(n_models, 1, figsize=(14, 5 * n_models))
        if n_models == 1:
            axes = [axes]

        for idx, model_name in enumerate(model_names):
            ax = axes[idx]

            # Plot distributions
            splits = ['test', 'ood_easy', 'ood_medium', 'ood_hard', 'ood_very_hard']
            colors = ['blue', 'green', 'orange', 'purple', 'red', 'brown']

            for split_name, color in zip(splits, colors):
                scores = self.results[model_name]['scores'].get(split_name)
                if scores is not None:
                    ax.hist(scores, bins=50, alpha=0.5, label=split_name, color=color, density=True)

            ax.set_xlabel('Score', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.set_title(f'{model_name.upper()} - Score Distributions', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        save_path = self.save_dir / "score_distributions.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
        plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train all density models on toy datasets")
    parser.add_argument("--dataset", type=str, required=True,
                       help="Path to toy dataset npz file")
    parser.add_argument("--save-dir", type=str, default="toy_experiments/results",
                       help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")

    args = parser.parse_args()

    # Run experiments
    trainer = DensityModelTrainer(
        dataset_path=args.dataset,
        save_dir=args.save_dir,
        device=args.device
    )

    trainer.run_all_experiments()

    print("\n" + "="*80)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Results saved to: {trainer.save_dir}")
    print("="*80)
