# Neural ODE Configuration Files

This directory contains configuration files for training Neural ODE models and performing OOD detection.

## Overview

The configs are organized into two categories:
1. **Training configs** (`*_train.yaml`): For training Continuous Normalizing Flow models
2. **OOD configs** (`*_ood.yaml`): For out-of-distribution detection using trained models

## Available Configurations

### Training Configurations

| Config File | Environment | Obs Dim | Action Dim | Dataset |
|-------------|-------------|---------|------------|---------|
| `hopper_train.yaml` | Hopper | 11 | 3 | hopper-medium-v2 |
| `halfcheetah_train.yaml` | HalfCheetah | 17 | 6 | halfcheetah-medium-v2 |
| `walker2d_train.yaml` | Walker2d | 17 | 6 | walker2d-medium-v2 |

### OOD Detection Configurations

| Config File | Environment | Trained Model | Dataset |
|-------------|-------------|---------------|---------|
| `hopper_ood.yaml` | Hopper | hopper/neural_ode/model.pt | hopper-medium-v2 |
| `halfcheetah_ood.yaml` | HalfCheetah | halfcheetah/neural_ode/model.pt | halfcheetah-medium-v2 |
| `walker2d_ood.yaml` | Walker2d | walker2d/neural_ode/model.pt | walker2d-medium-v2 |

## Quick Start

### 1. Train a Neural ODE Model

```bash
# Hopper
python neuralODE/neural_ode_density.py --config configs/neuralODE/hopper_train.yaml

# HalfCheetah
python neuralODE/neural_ode_density.py --config configs/neuralODE/halfcheetah_train.yaml

# Walker2d
python neuralODE/neural_ode_density.py --config configs/neuralODE/walker2d_train.yaml
```

### 2. Perform OOD Detection

```bash
# Hopper
python neuralODE/neural_ode_ood.py --config configs/neuralODE/hopper_ood.yaml

# HalfCheetah
python neuralODE/neural_ode_ood.py --config configs/neuralODE/halfcheetah_ood.yaml

# Walker2d
python neuralODE/neural_ode_ood.py --config configs/neuralODE/walker2d_ood.yaml
```

## Configuration Parameters

### Training Config Parameters

#### Dataset Configuration
- `npz`: Path to the dataset file (NPZ or pickle)
- `out`: Output directory for saving trained models

#### Training Parameters
- `epochs`: Number of training epochs (default: 200)
- `batch`: Batch size (default: 512)
- `lr`: Learning rate (default: 0.001)
- `wd`: Weight decay for regularization (default: 0.0)

#### Model Architecture
- `hidden_dims`: List of hidden layer dimensions (default: [512, 512])
- `activation`: Activation function - "silu" or "tanh" (default: "silu")
- `time_dependent`: Whether ODE is time-dependent (default: true)

#### ODE Solver Configuration
- `solver`: ODE solver algorithm (default: "dopri5")
  - Options: "dopri5", "rk4", "euler", "adams"
  - "dopri5": Adaptive Dormand-Prince (most accurate, slower)
  - "rk4": Fixed-step Runge-Kutta (balanced)
  - "euler": Simple Euler method (fastest, least accurate)
- `t0`: Start time for integration (default: 0.0)
- `t1`: End time for integration (default: 1.0)
- `rtol`: Relative tolerance for adaptive solvers (default: 1e-5)
- `atol`: Absolute tolerance for adaptive solvers (default: 1e-5)

#### Hardware & Logging
- `device`: "cuda" or "cpu" (default: "cuda")
- `log_every`: Log training progress every N steps (default: 100)
- `checkpoint_every`: Save checkpoint every N epochs (default: 50, 0 = disabled)
- `seed`: Random seed for reproducibility (default: 42)

### OOD Detection Config Parameters

#### Model & Data
- `model_path`: Path to trained Neural ODE model (required)
- `npz_path`: Path to NPZ file (leave empty for RL data)
- `data_path`: Path to RL dataset pickle file
- `task`: Task name (e.g., "hopper-medium-v2")

#### RL Data Parameters
- `obs_dim`: Observation dimension(s)
- `action_dim`: Action dimension

#### Data Splitting
- `val_ratio`: Validation split ratio (default: 0.2)
- `test_ratio`: Test split ratio (default: 0.2)

#### OOD Detection
- `anomaly_fraction`: Percentile for threshold setting (default: 0.01)
  - 0.01 = 1% lower percentile (strict)
  - 0.05 = 5% lower percentile (lenient)
- `batch_size`: Batch size for processing (default: 512)

#### Model Architecture (must match training)
- `hidden_dims`: Hidden layer dimensions
- `activation`: Activation function
- `time_dependent`: Time-dependent ODE flag
- `solver`: ODE solver
- `t0`, `t1`: Integration time bounds
- `rtol`, `atol`: Solver tolerances

#### Output
- `device`: "cuda" or "cpu"
- `plot_results`: Generate visualization plots (default: true)
- `save_dir`: Directory to save figures
- `save_model_path`: Path to save OOD model with threshold (optional)

## Customization

### Creating a Custom Config

To create a config for a new environment:

1. **Copy an existing config**:
   ```bash
   cp configs/neuralODE/hopper_train.yaml configs/neuralODE/myenv_train.yaml
   ```

2. **Update dataset paths**:
   ```yaml
   npz: "/path/to/myenv-dataset.pkl"
   out: "/path/to/output/myenv/neural_ode"
   ```

3. **Adjust dimensions** (if needed):
   - For RL environments, the input dimension is `obs_dim + action_dim`
   - This is automatically calculated from the dataset

4. **Tune hyperparameters**:
   - Larger `hidden_dims` for more complex datasets
   - Adjust `lr` if training is unstable
   - Increase `epochs` for better convergence

### Common Modifications

#### Faster Training (Less Accurate)
```yaml
solver: "rk4"  # or "euler"
rtol: 1.0e-3
atol: 1.0e-3
hidden_dims: [256, 256]  # Smaller network
batch: 1024  # Larger batches
```

#### Higher Accuracy (Slower)
```yaml
solver: "dopri5"
rtol: 1.0e-6
atol: 1.0e-6
hidden_dims: [1024, 1024]  # Larger network
batch: 256  # Smaller batches for stability
```

#### For Small Datasets
```yaml
batch: 128  # Smaller batch size
wd: 0.01  # Add weight decay for regularization
hidden_dims: [256, 256]  # Smaller network to prevent overfitting
```

#### For Large Datasets
```yaml
batch: 2048  # Larger batch size
hidden_dims: [1024, 1024, 1024]  # Deeper network
checkpoint_every: 20  # More frequent checkpoints
```

## Outputs

### Training Outputs

After training, the following files are saved to the `out` directory:

```
/public/gormpo/models/hopper/neural_ode/
├── model.pt                    # Final trained model
└── checkpoint_epoch_50.pt      # Checkpoint (if enabled)
```

### OOD Detection Outputs

After OOD detection, figures are saved to `save_dir`:

```
figures/neuralODE_ood/hopper/
├── neural_ode_roc_curve.png           # ROC curve with AUC
├── neural_ode_train_distribution.png  # Train/val distributions
└── neural_ode_ood_distribution.png    # OOD vs validation
```

## Environment-Specific Notes

### Hopper
- Observation dim: 11
- Action dim: 3
- Total input dim: 14
- Relatively simple dynamics, trains quickly

### HalfCheetah
- Observation dim: 17
- Action dim: 6
- Total input dim: 23
- More complex, may need more epochs or larger network

### Walker2d
- Observation dim: 17
- Action dim: 6
- Total input dim: 23
- Similar complexity to HalfCheetah

## Tips for Good Results

1. **Training**:
   - Monitor the NLL (negative log-likelihood) during training
   - NLL should decrease and stabilize
   - If NLL increases, reduce learning rate or add weight decay

2. **OOD Detection**:
   - Ensure model is well-trained (low validation NLL)
   - Adjust `anomaly_fraction` based on your tolerance for false positives
   - Lower values (e.g., 0.01) = stricter detection
   - Higher values (e.g., 0.05) = more lenient detection

3. **Performance**:
   - Use GPU for faster training (`device: "cuda"`)
   - Reduce `rtol`/`atol` for faster inference (with slight accuracy loss)
   - Use `batch_size: 1024` or larger for faster processing

## Troubleshooting

### Training is slow
- Use faster solver: `solver: "rk4"` or `solver: "euler"`
- Increase tolerances: `rtol: 1.0e-3, atol: 1.0e-3`
- Reduce network size: `hidden_dims: [256, 256]`

### NLL not decreasing
- Reduce learning rate: `lr: 0.0001`
- Check data normalization
- Try different activation: `activation: "tanh"`

### Out of memory
- Reduce batch size: `batch: 256`
- Reduce network size: `hidden_dims: [256, 256]`
- Use CPU: `device: "cpu"`

### Poor OOD detection
- Ensure model is well-trained
- Try different `anomaly_fraction` values
- Check that OOD data is actually different from training data
- Verify model architecture matches training config

## Additional Resources

- **Neural ODE Paper**: [arXiv:1806.07366](https://arxiv.org/abs/1806.07366)
- **FFJORD Paper**: [arXiv:1810.01367](https://arxiv.org/abs/1810.01367)
- **torchdiffeq Documentation**: [github.com/rtqichen/torchdiffeq](https://github.com/rtqichen/torchdiffeq)

## Examples

### Complete Workflow

```bash
# 1. Train the model
python neuralODE/neural_ode_density.py \
  --config configs/neuralODE/hopper_train.yaml

# 2. Perform OOD detection
python neuralODE/neural_ode_ood.py \
  --config configs/neuralODE/hopper_ood.yaml

# 3. Check results
ls figures/neuralODE_ood/hopper/
```

### Override Config Values

```bash
# Override specific parameters
python neuralODE/neural_ode_density.py \
  --config configs/neuralODE/hopper_train.yaml \
  --epochs 300 \
  --lr 0.0005 \
  --device cuda:1

# Override OOD parameters
python neuralODE/neural_ode_ood.py \
  --config configs/neuralODE/hopper_ood.yaml \
  --anomaly-fraction 0.05 \
  --batch-size 1024
```
