# KDE (Kernel Density Estimation) Module

This directory contains all KDE-related components for the MOPO project, including density estimation models, enhanced MBPO-KDE algorithms, dataset utilities, and configuration files.

## Quick GORMPO training with trained density estimators

KDE

```
python mopo.py --config configs/kde/gormpo_halfcheetah_medium_expert_sparse.yaml --devid 0
```

VAE:

```
python mopo.py --config configs/vae/gormpo_halfcheetah_medium_expert_sparse.yaml --devid 0
```

RealNVP:

```
python mopo.py --config configs/realnvp/gormpo_halfcheetah_medium_expert_sparse.yaml --devid 0
```

Diffusion: 
```
python mopo.py --config configs/diffusion/halfcheetah_medium_expert_sparse.yaml --devid 0
```

NeuralODE
```
python mopo.py --config configs/neuralODE/halfcheetah_medium_expert_sparse.yaml --devid 0
```

Configs *must* include classifier_model_name, task, data_path (for sparse saved datasets), algo-name, reward-penalty-coef, penalty_type parameters.


For hyperparameter search, create a bash script like 'bash_scr/halfcheetah_medium_expert_v2_gormpo_kde_sparse3.sh' by specifying the *config*. Then run as follows:

```
chmod +x /bash_scr/halfcheetah_medium_expert_v2_gormpo_kde_sparse3.sh 
./bash_scr/halfcheetah_medium_expert_v2_gormpo_kde_sparse3.sh
```

## Core Components

### 1. RealNVP Normalizing Flow (`realnvp.py`)

A complete implementation of the RealNVP normalizing flow model for density estimation:

**Features:**
- Invertible neural networks with coupling layers
- Automatic threshold selection for anomaly detection
- Comprehensive training with early stopping
- Built-in evaluation metrics (ROC-AUC, accuracy)
- Synthetic data generation for testing

**Usage:**
```python
from KDE.realnvp import RealNVP
import torch

# Create model
model = RealNVP(input_dim=10, num_layers=6, hidden_dims=[256, 256])

# Train with automatic validation split
history = model.fit(train_data, val_data, epochs=100)

# Detect anomalies
anomalies = model.predict_anomaly(test_data)
```

### 2. MBPO-KDE Training (`mopo.py`, `train.py`)

Enhanced model-based offline RL with KDE-based uncertainty estimation:

**Key Improvements:**
- KDE-based world model uncertainty quantification
- Weighted sampling strategies for model rollouts
- Advanced penalty mechanisms (linear, inverse, exponential, softplus)
- Integration with Abiomed medical device environments

**Configuration Options:**
- `reward-penalty-coef`: Penalty coefficient for model uncertainty
- `penalty_type`: Type of penalty function ("linear", "inverse", "exponential", "softplus")
- `classifier_model_name`: Pre-trained KDE classifier for uncertainty estimation
- `gamma1`, `gamma2`, `gamma3`: Environment-specific reward shaping parameters

### 3. Dataset Loading Utilities

Robust dataset loading with automatic validation split creation:

**Test the utilities:**
```bash
cd KDE
python test_dataset_loader.py    # Run comprehensive tests
python example_usage.py          # See usage examples
```

**Features:**
- Unified interface for pickle, npz, and environment datasets
- Automatic train/validation splits when needed
- Comprehensive error handling and validation
- Support for Abiomed's pre-split datasets

## Running Experiments

### Quick Start

1. **Basic MBPO-KDE training:**
```bash
python KDE/mopo.py --algo-name mbpo_kde --reward-penalty-coef 0.005 --epoch 100
```

2. **Using configuration files:**
```bash
python KDE/mopo.py --config KDE/configs/mbpo_kde_ws.yaml
```

3. **Running batch experiments:**
```bash
bash KDE/scripts/run_mbpo_kde.sh
```

### Configuration Files

- `mbpo_kde.yaml` - Basic MBPO-KDE configuration
- `mbpo_kde_acp.yaml` - With ACP (Action Conditional Penalty)
- `mbpo_kde_ws.yaml` - With weighted sampling
- `mbpo_kde_ws_exp.yaml` - Exponential penalty variant
- `mbpo_kde_ws_inverse.yaml` - Inverse penalty variant
- `mbpo_kde_ws_soft.yaml` - Softplus penalty variant

### Key Parameters

**Model Parameters:**
- `n-ensembles`: Number of ensemble models (default: 7)
- `n-elites`: Number of elite models (default: 5)
- `rollout-length`: Length of model rollouts (default: 5)
- `rollout-batch-size`: Batch size for rollouts (default: 10000)

**KDE Parameters:**
- `reward-penalty-coef`: Uncertainty penalty coefficient (0.0-1.0)
- `penalty_type`: Penalty function type
- `classifier_model_name`: Path to pre-trained KDE classifier

**Environment Parameters:**
- `gamma1`, `gamma2`, `gamma3`: Reward shaping coefficients
- `noise_rate`, `noise_scale`: Environment noise parameters

## Integration with Main Codebase

The KDE module integrates with the main MOPO codebase through:

1. **Common utilities** - Uses shared `common/util.py` for logging, device management
2. **Model architectures** - Extends `models/` policy and transition models
3. **Training infrastructure** - Built on shared `trainer.py` and `algo/` components
4. **Environment support** - Compatible with existing environment wrappers

## Testing and Validation

### Dataset Loading Tests
```bash
cd KDE
python test_dataset_loader.py
```
Tests cover:
- Pickle and NPZ file loading
- Environment dataset integration
- Abiomed dataset handling
- Validation split creation
- Error handling and edge cases

### Model Training Tests
```bash
cd KDE
python realnvp.py  # Run built-in RealNVP tests
```

## Advanced Usage

### Custom Density Models

Extend the RealNVP implementation for specific use cases:

```python
# Custom coupling layer configuration
model = RealNVP(
    input_dim=state_action_dim,
    num_layers=8,
    hidden_dims=[512, 512, 256],
    device='cuda'
)

# Train on combined state-action space
combined_data = np.concatenate([observations, actions], axis=1)
history = model.fit(train_data, val_data, epochs=200, lr=1e-4)

# Use for uncertainty quantification in RL
uncertainty_scores = -model.log_prob(new_data)
```

### Custom Penalty Functions

Implement new penalty types in `transition_model.py`:

```python
def custom_penalty(uncertainty, coef):
    """Custom uncertainty penalty function."""
    return coef * torch.sigmoid(uncertainty - threshold)
```

## Performance Notes

- **GPU Usage**: All models support CUDA acceleration
- **Memory**: Large ensemble models may require significant GPU memory
- **Training Time**: MBPO-KDE training typically takes 2-4x longer than standard MBPO
- **Hyperparameter Sensitivity**: KDE methods can be sensitive to `reward-penalty-coef`

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure you're running from the repository root directory
2. **CUDA Errors**: Check GPU memory availability for ensemble models
3. **Convergence Issues**: Try adjusting `reward-penalty-coef` (start with 0.001-0.01)
4. **Dataset Loading**: Verify file paths and format compatibility

### Debug Mode

Enable verbose logging:
```bash
python KDE/mopo.py --config KDE/configs/mbpo_kde.yaml --verbose
```

## Citation

If you use this KDE implementation, please cite:

- Dinh et al. (2017) "Density estimation using Real NVP"
- Yu et al. (2020) "MOPO: Model-based Offline Policy Optimization"

## Contributing

When adding new KDE-related features:

1. Place new models in the `KDE/` directory
2. Add corresponding config files to `KDE/configs/`
3. Update this README with usage examples
4. Add tests to verify functionality
5. Update import paths to work from the KDE directory