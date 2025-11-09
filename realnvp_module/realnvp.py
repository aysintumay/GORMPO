import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import argparse
import yaml
import os
import sys
import pickle
import seaborn as sns
import gym
# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common.buffer import ReplayBuffer
from common.util import load_dataset_with_validation_split


class MLP(nn.Module):
    """Multi-layer perceptron for coupling layer transformations."""

    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class CouplingLayer(nn.Module):
    """RealNVP coupling layer with affine transformation."""

    def __init__(self, input_dim: int, hidden_dims: List[int], mask: torch.Tensor):
        super().__init__()

        self.register_buffer('mask', mask)
        masked_dim = int(mask.sum().item())

        # Scale and translation networks
        self.scale_net = MLP(masked_dim, hidden_dims, input_dim - masked_dim).to(self.mask.device)
        self.translate_net = MLP(masked_dim, hidden_dims, input_dim - masked_dim).to(self.mask.device)

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through coupling layer.

        Args:
            x: Input tensor
            reverse: If True, compute inverse transformation

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.mask.device)
        x_masked = x * self.mask
        x_unmasked = x * (1 - self.mask)

        # Get scale and translation from masked components
        x_masked_input = x_masked[:, self.mask.bool()]
        scale = self.scale_net(x_masked_input)
        translate = self.translate_net(x_masked_input)

        if not reverse:
            # Forward transformation: y = x * exp(s) + t
            log_scale = torch.tanh(scale)  # Stabilize scale
            x_unmasked_vals = x_unmasked[:, ~self.mask.bool()]
            y_unmasked = x_unmasked_vals * torch.exp(log_scale) + translate

            # Reconstruct full tensor
            y = x.clone()
            y[:, ~self.mask.bool()] = y_unmasked

            log_det = log_scale.sum(dim=1)

        else:
            # Inverse transformation: x = (y - t) * exp(-s)
            log_scale = torch.tanh(scale)
            x_unmasked_vals = x_unmasked[:, ~self.mask.bool()]
            y_unmasked = (x_unmasked_vals - translate) * torch.exp(-log_scale)

            # Reconstruct full tensor
            y = x.clone()
            y[:, ~self.mask.bool()] = y_unmasked

            log_det = -log_scale.sum(dim=1)

        return y, log_det


class RealNVP(nn.Module):
    """RealNVP normalizing flow model for density estimation."""

    def __init__(
        self,
        input_dim: int =2,
        num_layers: int = 6,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cpu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.num_layers = num_layers
        self.device = device

        # Create alternating masks
        self.masks = []
        for i in range(num_layers):
            mask = torch.zeros(input_dim)
            if i % 2 == 0:
                mask[::2] = 1  # Even indices
            else:
                mask[1::2] = 1  # Odd indices
            self.masks.append(mask)

        # Create coupling layers
        self.coupling_layers = nn.ModuleList([
            CouplingLayer(input_dim, hidden_dims, mask.to(device))
            for mask in self.masks
        ])

        # Prior distribution parameters (standard normal)
        self.register_buffer('prior_mean', torch.zeros(input_dim))
        self.register_buffer('prior_std', torch.ones(input_dim))

        # Threshold for anomaly detection
        self.threshold = None

    def _apply(self, fn):
        """Override _apply to update self.device when model is moved."""
        super()._apply(fn)
        # Update self.device to match the actual device of parameters
        if len(list(self.parameters())) > 0:
            self.device = next(self.parameters()).device
        return self

    def forward(self, x: torch.Tensor, reverse: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the flow.

        Args:
            x: Input tensor of shape (batch_size, input_dim)
            reverse: If True, sample from the model

        Returns:
            Transformed tensor and log determinant of Jacobian
        """
        # Use the actual device of model parameters instead of self.device
        model_device = next(self.parameters()).device
        log_det_total = torch.zeros(x.shape[0], device=model_device)

        if not reverse:
            # Forward: data -> latent
            z = x
            for layer in self.coupling_layers:
                z, log_det = layer(z, reverse=False)
                log_det_total += log_det
        else:
            # Reverse: latent -> data
            z = x
            for layer in reversed(self.coupling_layers):
                z, log_det = layer(z, reverse=True)
                log_det_total += log_det

        return z, log_det_total

    def score_samples(self, x: torch.Tensor) -> torch.Tensor:
        """Compute log probability of data points."""
        z, log_det = self.forward(x, reverse=False)

        # Log probability under prior (standard normal)
        log_prior = -0.5 * (
            z.pow(2).sum(dim=1) +
            self.input_dim * np.log(2 * np.pi)
        )
        # print(log_det.max().item(), np.exp(log_det.max().item()))

        return (log_prior + log_det).cpu()

    def sample(self, num_samples: int) -> torch.Tensor:
        """Generate samples from the model."""
        with torch.no_grad():
            # Sample from prior
            model_device = next(self.parameters()).device
            z = torch.randn(num_samples, self.input_dim, device=model_device)

            # Transform to data space
            x, _ = self.forward(z, reverse=True)

        return x

    def fit(
        self,
        train_data: torch.Tensor,
        val_data: torch.Tensor,
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 1e-3,
        patience: int = 15,
        verbose: bool = True
    ) -> dict:
        """
        Train the RealNVP model.

        Args:
            train_data: Training dataset
            val_data: Validation dataset for threshold selection
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            patience: Early stopping patience
            verbose: Print training progress

        Returns:
            Training history dictionary
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=patience//2
        )

        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_data),
            batch_size=batch_size,
            shuffle=True
        )

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('-inf')
        patience_counter = 0

        self.train()
        val_data = (val_data + 1e-3)
        for epoch in range(epochs):
            train_loss = 0.0
            num_batches = 0

            for batch_data, in train_loader:
                batch_data = batch_data.to(self.device)
                # noise = torch.rand_like(batch_data)*0.01   # uniform [0,1)
                batch_data = (batch_data + 1e-3)
                optimizer.zero_grad()

                # Compute negative log likelihood
                log_prob = self.score_samples(batch_data)
                loss = -log_prob.mean()

                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            train_loss /= num_batches

            # Validation
            self.eval()
            with torch.no_grad():
                # noise = torch.rand_like(val_data)*0.01  # uniform [0,1)
                
                val_log_prob = self.score_samples(val_data.to(self.device))
                val_loss = -val_log_prob.mean().item()
                # print("VALIDATION", val_log_prob.max(), np.exp(val_log_prob.max().item()))
            self.train()

            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            if verbose and epoch % 5 == 0:
                print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')

            # Early stopping
            if val_loss > best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f'Early stopping at epoch {epoch}')
                break

        # Set threshold based on validation data
        self.set_threshold(val_data)

        return history

    def set_threshold(
        self,
        val_data: torch.Tensor,
        anomaly_fraction: float = 0.01
    ):
        """
        Set threshold for anomaly detection based on validation data.

        Args:
            val_data: Validation dataset (assumed to be normal data)
            anomaly_fraction: Fraction of validation data to classify as anomalies
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.score_samples(val_data.to(self.device))

        # Set threshold as percentile of validation log probabilities
        self.threshold = torch.quantile(log_probs, anomaly_fraction).item()

        print(f'Threshold set to {self.threshold:.4f} '
              f'(marking {anomaly_fraction*100:.1f}% of validation data as anomalies)')

    def predict_anomaly(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict anomalies based on log probability threshold.

        Args:
            x: Input data

        Returns:
            Boolean tensor indicating anomalies (True = anomaly)
        """
        if self.threshold is None:
            raise ValueError("Threshold not set. Call set_threshold() first.")

        self.eval()
        with torch.no_grad():
            log_probs = self.score_samples(x.to(self.device))

        return log_probs < self.threshold
    
    def predict(self, X):
        """
        Predict anomalies based on threshold

        Args:
            X: Test data

        Returns:
            predictions: 1 for normal, -1 for anomaly
        """
        scores = self.score_samples(X)
        return np.where(scores.cpu() >= self.threshold, 1, -1)
    
    def evaluate_anomaly_detection(
    self,
    normal_data: torch.Tensor,
    anomaly_data: torch.Tensor,
    plot: bool = True) -> dict:
        """
        Evaluate anomaly detection performance.
        """
        self.eval()

        # >>> Get the true device of the model <<<
        model_device = next(self.parameters()).device

        # >>> Move data to the same device as the model <<<
        normal_data = normal_data.to(model_device)
        anomaly_data = anomaly_data.to(model_device)

        print("Model device:", model_device)
        print("Normal data device:", normal_data.device)

        with torch.no_grad():
            normal_log_probs = self.score_samples(normal_data).cpu().numpy()
            anomaly_log_probs = self.score_samples(anomaly_data).cpu().numpy()

        # Create labels (0 = normal, 1 = anomaly)
        y_true = np.concatenate([
            np.zeros(len(normal_log_probs)),
            np.ones(len(anomaly_log_probs))
        ])

        # Use negative log prob as anomaly score
        y_scores = np.concatenate([-normal_log_probs, -anomaly_log_probs])

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Compute accuracy with current threshold
        if self.threshold is not None:
            predictions = np.concatenate([
                normal_log_probs < self.threshold,
                anomaly_log_probs < self.threshold
            ])
            accuracy = (predictions == y_true).mean()
        else:
            accuracy = None

        results = {
            'roc_auc': roc_auc,
            'accuracy': accuracy,
            'normal_log_prob_mean': normal_log_probs.mean(),
            'normal_log_prob_std': normal_log_probs.std(),
            'anomaly_log_prob_mean': anomaly_log_probs.mean(),
            'anomaly_log_prob_std': anomaly_log_probs.std()
        }

        if plot:
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve for Anomaly Detection')
            plt.legend()
            plt.grid(True)
            plt.show()

        return results

    def save_model(self, save_path: str):
        """
        Save the RealNVP model and metadata.

        Args:
            save_path: Base path for saving (without extension)
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save model state dict
        torch.save(self.state_dict(), f"{save_path}_model.pth")
        #calculate the logprobs in training data
        self.eval()
        with torch.no_grad():
            train_log_probs = self.score_samples(train_data.to(self.device))
        # Save metadata (threshold and config)
        metadata = {
            'threshold': self.threshold,
            'input_dim': self.input_dim,
            'num_layers': self.num_layers,
            'device': self.device,
            "mean":train_log_probs.cpu().mean().item(),
            "std":train_log_probs.cpu().std().item()

        }

        with open(f"{save_path}_meta_data.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Model saved to: {save_path}_model.pth")
        print(f"Metadata saved to: {save_path}_meta_data.pkl")

    @classmethod
    def load_model(cls, save_path: str, hidden_dims: List[int] = [256, 256]):
        """
        Load a saved RealNVP model.

        Args:
            save_path: Base path for loading (without extension)
            hidden_dims: Hidden layer dimensions (must match saved model)

        Returns:
            Loaded RealNVP model
        """
        # Load metadata
        with open(f"{save_path}_meta_data.pkl", 'rb') as f:
            metadata = pickle.load(f)

        # Create model with saved configuration
        model = cls(
            input_dim=metadata['input_dim'],
            num_layers=metadata['num_layers'],
            hidden_dims=hidden_dims,
            device=metadata['device']
        )

        # Load model state dict
        model.load_state_dict(torch.load(f"{save_path}_model.pth", map_location=metadata['device']))
        # model.to(cls.device)

        # Restore threshold
        model.threshold = metadata['threshold']

        print(f"Model loaded from: {save_path}_model.pth")
        print(f"Metadata loaded from: {save_path}_meta_data.pkl")
        print(f"Threshold: {model.threshold}")
        model_dict = {'model': model, 'thr': model.threshold, 'mean': metadata["mean"], 'std': metadata["std"]}
        return model_dict


def create_synthetic_data(n_samples=1000, dim=2, anomaly_type="outlier"):
    """
    Generate synthetic normal and anomalous data in arbitrary dimensions.

    Args:
        n_samples (int): number of normal samples
        dim (int): dimensionality of data
        anomaly_type (str): "outlier" or "uniform"

    Returns:
        (torch.FloatTensor, torch.FloatTensor): normal_data, anomaly_data
    """
    normal_data = []
    for _ in range(n_samples):
        if np.random.rand() < 0.7:
            # Main cluster around 0
            mean = np.zeros(dim)
            cov = np.eye(dim)                      # identity covariance
            sample = np.random.multivariate_normal(mean, cov, 1)
        else:
            # Secondary cluster around 3
            mean = np.ones(dim) * 3
            cov = 0.5 * np.eye(dim)                # smaller spread
            sample = np.random.multivariate_normal(mean, cov, 1)
        normal_data.append(sample[0])

    normal_data = np.array(normal_data)

    # Anomalous data
    if anomaly_type == "outlier":
        mean = np.ones(dim) * 10
        cov = 2 * np.eye(dim)
        anomaly_data = np.random.multivariate_normal(mean, cov, n_samples // 5)
    elif anomaly_type == "uniform":
        anomaly_data = np.random.uniform(-5, 8, (n_samples // 5, dim))
    else:
        raise ValueError(f"Unknown anomaly type: {anomaly_type}")

    return torch.FloatTensor(normal_data), torch.FloatTensor(anomaly_data)


def load_rl_data_for_kde(args, env=None, val_split_ratio=0.2, test_split_ratio=0.2):
    """
    Load RL dataset and prepare next_observations + actions for KDE training.

    Args:
        args: Arguments containing data_path, obs_shape, action_dim
        env: Environment object (for Abiomed datasets)
        val_split_ratio: Fraction of data for validation

    Returns:
        Tuple of (train_data, val_data, kde_input_dim)
    """
    # Load dataset using the existing utility function
    dataset_result = load_dataset_with_validation_split(
        args=args,
        env=env,
        val_split_ratio=val_split_ratio,
        test_split_ratio=test_split_ratio
    )

    train_dataset = dataset_result['train_data']
    val_dataset = dataset_result['val_data']
    test_dataset = dataset_result['test_data']
    buffer_len = dataset_result['buffer_len']

    # print(f"Loaded dataset with {buffer_len} total samples")

    # Initialize ReplayBuffer
   
    offline_buffer = ReplayBuffer(
            buffer_size=len(train_dataset["observations"]),
            obs_shape=args.obs_dim,
            obs_dtype=np.float32,
            action_dim=args.action_dim,
            action_dtype=np.float32
            )
    for i in range(len(train_dataset['observations'])):
        offline_buffer.add(
            obs=train_dataset['observations'][i],
            next_obs=train_dataset['next_observations'][i],
            action=train_dataset['actions'][i],
            reward=train_dataset['rewards'][i],
            terminal=train_dataset['terminals'][i]
        )

    # Extract concatenated next_observations + actions for KDE
    train_next_obs = torch.FloatTensor(train_dataset['next_observations'])
    train_actions = torch.FloatTensor(train_dataset['actions'])

    # Concatenate next_observations and actions for training data
    train_kde_input = torch.cat([train_next_obs, train_actions], dim=1)

   
    val_next_obs = torch.FloatTensor(val_dataset['next_observations'])
    val_actions = torch.FloatTensor(val_dataset['actions'])

 
    test_next_obs = torch.FloatTensor(test_dataset['next_observations'])
    test_actions = torch.FloatTensor(test_dataset['actions'])
    # Concatenate next_observations and actions for validation data
    val_kde_input = torch.cat([val_next_obs, val_actions], dim=1)
    test_kde_input = torch.cat([test_next_obs, test_actions], dim=1)

    # Calculate input dimension for KDE model
    kde_input_dim = train_kde_input.shape[1]

    print(f"KDE input shape: {train_kde_input.shape}")
    print(f"KDE input dimension: {kde_input_dim}")
    print(f"Next observations dim: {train_next_obs.shape[1]}, Actions dim: {train_actions.shape[1]}")

    return train_kde_input, val_kde_input, test_kde_input, kde_input_dim


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def parse_args():
    """Parse command line arguments."""
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="configs/realnvp/hopper.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    parser = argparse.ArgumentParser(parents=[config_parser])
   
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu). Overrides config file.')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs. Overrides config file.')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output. Overrides config file.')

    # RL dataset specific arguments
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to RL dataset file (pickle or npz)')
    parser.add_argument('--task', type=str, default='hopper-medium-v2',
                        help='Task name (e.g., abiomed, halfcheetah-medium-v2)')
    parser.add_argument('--obs_dim', type=int, nargs='+', default=[11],
                        help='Observation shape (default: [17] for HalfCheetah)')
    parser.add_argument('--action_dim', type=int, default=3,
                        help='Action dimension (default: 6 for HalfCheetah)')
    parser.set_defaults(**config)
    args = parser.parse_args(remaining_argv)
    args.config = config
   
    return args

def plot_likelihood_distributions(
    model,
    train_data,
    val_data,
    ood_data=None,
    thr= None,
    title="Likelihood Distribution",
    savepath=None,
    bins=50
):
    """
    Visualize log-likelihood distributions for train, val, and OOD data.

    Args:
        model: density model with .score_samples(X) method (returns log probs)
        train_data: np.ndarray or torch.Tensor, in-distribution training set
        val_data:   np.ndarray or torch.Tensor, held-out validation set
        ood_data:   np.ndarray or torch.Tensor, optional OOD dataset
        title: str, title for the plot
        savepath: str, optional path to save figure
        bins: int, number of histogram bins
    """
    # --- Compute log-likelihoods ---
    logp_train = model.score_samples(train_data).detach().cpu().numpy()
    logp_val   = model.score_samples(val_data).detach().cpu().numpy()
    logp_ood   = None
    if ood_data is not None:
        logp_ood = model.score_samples(ood_data).detach().cpu().numpy()
   
    
    # --- Plot ---
    plt.figure(figsize=(8, 5))
    sns.histplot(logp_train, bins=bins, color="blue", alpha=0.4, label="Train", kde=True)
    sns.histplot(logp_val, bins=bins, color="green", alpha=0.4, label="Validation", kde=True)
    plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')
    plt.xlabel("Log-likelihood")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()
    plt.savefig(f"figures/train_distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/train_distribution_kde.png")
    plt.figure(figsize=(8, 5))
    

    if logp_ood is not None:
        sns.histplot(logp_ood, bins=bins, color="red", alpha=0.4, label="Test", kde=True)
        plt.axvline(x=thr, color='tab:red', linestyle='--', label='Threshold')

    plt.xlabel("Log-likelihood")
    plt.ylabel("Frequency")
    plt.title(title)
    plt.legend()

    plt.savefig(f"figures/distribution_kde.png", dpi=300, bbox_inches="tight")
    print(f"Saved figure at figures/distribution_kde.png")

def create_ood_test(data, model, percentage=[0.1,0.3,0.5,0.7,0.9]):

   
    data_dict = {}
    res_dict = {}
    predictions_tr = model.predict(data)
    small_data = data[predictions_tr == 1][np.random.choice(len(data), int(0.2 * len(data)), replace=False)].cpu().numpy()

    for perc in percentage:
        base_test = small_data[predictions_tr == 1][np.random.choice(len(small_data), int(perc * len(small_data)), replace=False)].cpu().numpy()

        small_train = small_data[predictions_tr == 1][np.random.choice(len(small_data), int((1-perc) * len(small_data)), replace=False)].cpu().numpy()
        noisy_train = small_train + np.random.normal(0, 0.1, small_train.shape)
        data_dict[perc] = torch.FloatTensor(np.concatenate([base_test, noisy_train], axis=0)).to(model.device)
        # predictions_test = model.predict(data_dict[perc])
        scores_test = model.score_samples(data_dict[perc])
        res_dict[perc] = scores_test.mean().item()
        print(f"Percentage: {perc}, Mean Score: {scores_test.mean().item():.4f}")
    return data_dict,res_dict

def plot_tsne(tsne_data1, preds, title):

    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_data1[preds == 1, 0], tsne_data1[preds == 1, 1], 
                color='blue', label='ID', alpha=0.5)
    plt.scatter(tsne_data1[preds == -1, 0], tsne_data1[preds == -1, 1], 
                color='red', label='OOD', alpha=0.5)
    plt.title(title)
    
    plt.xlabel('TSNE Dimension 1')
    plt.ylabel('TSNE Dimension 2')

    #save figure
    plt.legend()
    plt.savefig(f"figures/{title.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    args.obs_dim = (args.obs_dim,)

    # Load configuration
    print(f"Loading configuration from: {args.config}")
    config = args.config

    # Override config with command line arguments if provided
    if args.device is not None:
        config['device'] = args.device
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.verbose:
        config['verbose'] = True

    # Override config with RL-specific arguments
    if args.data_path is not None:
        args.data_path = args.data_path  # Keep as args attribute for compatibility
    if args.task != 'synthetic':
        args.task = args.task
    args.obs_shape = tuple(args.obs_dim)
    args.action_dim = args.action_dim

    # Set random seed for reproducibility
    if 'seed' in config:
        torch.manual_seed(config['seed'])
        np.random.seed(config['seed'])

    # Determine device: CLI overrides config
    cli_device = args.device
    cfg_device = config.get('device', None)

    device = cli_device if cli_device is not None else (cfg_device or 'cpu')

    if isinstance(device, str) and device.startswith("cuda") and not torch.cuda.is_available():
        print(f"{device} not available, falling back to CPU")
        device = "cpu"

    # Make sure everyone agrees on the same device
    args.device = device
    config["device"] = device

    print(f"Using device: {device}")


    # Choose data loading mode
    use_rl_data = config.get('use_rl_data', args.task != 'synthetic')

    # Only create environment if needed (when data_path is not provided)
    env = None
    if use_rl_data and args.data_path is None:
        try:
            env = gym.make(args.task)
        except Exception as e:
            print(f"Warning: Could not create environment {args.task}: {e}")
            print("Will use data_path instead if provided")

    if use_rl_data:
        print("Loading RL dataset for KDE training...")

        # Load RL data (next_observations + actions)
        train_data, val_data, test_data, kde_input_dim = load_rl_data_for_kde(
            args=args,
            env=env,  
            val_split_ratio=config.get('val_ratio', 0.2),
            test_split_ratio=config.get('test_ratio', 0.2)
        )

        # For RL data, we don't have separate anomaly data for evaluation
        # We'll use a portion of validation data as "normal" and generate synthetic anomalies
        n_test = len(test_data)
        test_normal = test_data

        # Generate synthetic anomalies in the same dimension as RL data
        anomaly_data = torch.randn(n_test // 2, kde_input_dim) * 3 + 5  # Offset anomalies

        # Update input dimension in config
        config['input_dim'] = kde_input_dim

    else:
        print("Creating synthetic data...")
        normal_data, anomaly_data = create_synthetic_data(
            n_samples=config.get('n_samples', 2000),
            dim=config.get('input_dim', 2),
            anomaly_type=config.get('anomaly_type', 'outlier')
        )

        # # Split normal data into train/val/test
        # train_ratio = config.get('train_ratio', 0.6)
        # val_ratio = config.get('val_ratio', 0.2)

        # n_train = int(train_ratio * len(normal_data))
        # n_val = int(val_ratio * len(normal_data))

        # train_data = normal_data[:n_train]
        # val_data = normal_data[n_train:n_train+n_val]
        # test_normal = normal_data[n_train+n_val:]

    if config.get('verbose', True):
        print(f"Data shapes - Train: {train_data.shape}, Val: {val_data.shape}, "
              f"Test Normal: {test_normal.shape}, Test Anomaly: {anomaly_data.shape}")

    # Create and train model
    print("Creating RealNVP model...")
    model = RealNVP(
        input_dim=config.get('input_dim', 2),
        num_layers=config.get('num_layers', 6),
        hidden_dims=config.get('hidden_dims', [256, 256]),
        device=device
    ).to(device)

    # print("Training RealNVP model...")
    # history = model.fit(
    #     train_data=train_data,
    #     val_data=val_data,
    #     epochs=config.get('epochs', 100),
    #     batch_size=config.get('batch_size', 128),
    #     lr=config.get('lr', 1e-3),
    #     patience=config.get('patience', 15),
    #     verbose=config.get('verbose', True)
    # )
    # # Save model if requested
    # if config.get('model_save_path', False):
    #     save_path = config.get('model_save_path', 'saved_models/realnvp')
    #     os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #     model.save_model(save_path)
    #     print(f"Model saved to: {save_path}_model.pth")
    #load pretrained model
    model_dict = RealNVP.load_model(save_path=args.model_save_path)
    model = model_dict['model']
    model.to(device)

    # print(args.device)
    model.threshold = model_dict['thr']
    # print(model.device)
    # Evaluate anomaly detection
    print("\nEvaluating anomaly detection performance...")
    results = model.evaluate_anomaly_detection(
        normal_data=test_normal.to(model.device),
        anomaly_data=anomaly_data.to(model.device),
        plot=config.get('plot_results', True)
    )
    print(type(train_data))
    train_data = train_data.to(model.device)
    val_data = val_data.to(model.device)
    test_data = test_data.to(model.device)
    print(train_data.device)
    print(model.device)

    predictions_tr = model.predict(train_data)
    scores_tr = model.score_samples(train_data)
    print('TRAINING SCORE', scores_tr.mean().item())
    scores_test_in_dist = model.score_samples(test_data)
    anomaly_test_res  = model.score_samples(anomaly_data.to(model.device))

    # small_train = train_data[predictions_tr == 1][: int(0.1 * len(train_data))].cpu().numpy()
    # noisy_train = small_train + np.random.normal(0, 0.1, small_train.shape)
    # normal_data = torch.FloatTensor(np.concatenate([small_train, noisy_train], axis=0)).to(model.device)
    test_ood_dict,res_ood_dict = create_ood_test(train_data, model, percentage=[0.1,0.3,0.5,0.7])

    # predictions_test = model.predict(normal_data)
   
    # scores_test = model.score_samples(normal_data)
    # print("Number of data with likelihood>0",(scores_test>0).sum())
    # print("Scores test OOD", scores_test.mean().item())
    print("Scores test ID", scores_test_in_dist.mean().item())
    large_test_dict = {0.0: scores_test_in_dist.mean().item(),
                       0.1: res_ood_dict[0.1],
                          0.3: res_ood_dict[0.3],
                            0.5: res_ood_dict[0.5],
                                0.7: res_ood_dict[0.7],
                                1.0: anomaly_test_res.mean().item()
                       }
    # anomaly_count = (np.array(predictions_test) == -1).sum()
    # print("Max density score", scores_test.max().item()) 
    # print("Min density score", scores_test.min().item())

    # print(f"Test data anomalies detected: {anomaly_count}/{len(normal_data)} ({(anomaly_count/len(normal_data)):.1%})")

    # plot_likelihood_distributions(
    #                     model,
    #                     train_data,
    #                     val_data,
    #                     ood_data=normal_data,
    #                     thr= model.threshold,
    #                     title="Likelihood Distribution",
    #                     savepath=None,
    #                     bins=50
    #                 )
    print(f"ROC AUC: {results['roc_auc']:.3f}")
    print(f"Accuracy: {results['accuracy']:.3f}")
    print(f"Normal data log prob: {results['normal_log_prob_mean']:.3f} ± {results['normal_log_prob_std']:.3f}")
    print(f"Anomaly data log prob: {results['anomaly_log_prob_mean']:.3f} ± {results['anomaly_log_prob_std']:.3f}")


    

    print("\nRealNVP training and evaluation completed!")