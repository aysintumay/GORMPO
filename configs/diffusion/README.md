# Diffusion Model OOD Detection Configuration Files

This directory contains configuration files for Diffusion model out-of-distribution (OOD) detection.

## Available Configurations

| Config File | Environment | Model Directory | Dataset |
|-------------|-------------|-----------------|---------|
| `hopper_ood.yaml` | Hopper | /public/gormpo/models/hopper/diffusion | hopper-medium-v2 |
| `halfcheetah_ood.yaml` | HalfCheetah | /public/gormpo/models/halfcheetah/diffusion | halfcheetah-medium-v2 |
| `walker2d_ood.yaml` | Walker2d | /public/gormpo/models/walker2d/diffusion | walker2d-medium-v2 |

## Quick Start

```bash
# Hopper
python diffusion/diffusion_ood.py --config configs/diffusion/hopper_ood.yaml

# HalfCheetah
python diffusion/diffusion_ood.py --config configs/diffusion/halfcheetah_ood.yaml

# Walker2d
python diffusion/diffusion_ood.py --config configs/diffusion/walker2d_ood.yaml
```

## Configuration Parameters

### Model & Data
- `model_dir`: Directory containing `checkpoint.pt` and `scheduler/` subdirectory (required)
- `data_path`: Path to RL dataset file (NPZ or pickle format)

### Data Splitting
- `val_ratio`: Validation split ratio (default: 0.2)
- `test_ratio`: Test split ratio (default: 0.2)

### OOD Detection
- `anomaly_fraction`: Percentile for threshold setting (default: 0.01)
  - 0.01 = 1% lower percentile (strict)
  - 0.05 = 5% lower percentile (lenient)
- `batch_size`: Batch size for ELBO computation (default: 256)
  - Diffusion is memory-intensive, use smaller batches than Neural ODE
  - Reduce to 128 or 64 if you encounter OOM
- `max_samples`: Maximum samples to load (default: 10000)
  - Limits data loading for faster evaluation
  - Reduce to 5000 or 3000 for quicker results

### Hardware
- `device`: "cuda" or "cpu" (default: "cuda")

### Output
- `plot_results`: Generate visualization plots (default: true)
- `save_dir`: Directory to save figures
- `save_model_path`: Path to save OOD model metadata (optional)

## Configuration Structure Example

```yaml
# Model directory (must contain checkpoint.pt and scheduler/)
model_dir: "/public/gormpo/models/hopper/diffusion"

# Data configuration
data_path: "/public/d4rl/hopper-medium-v2.pkl"

# Data splitting
val_ratio: 0.2
test_ratio: 0.2

# OOD detection parameters
anomaly_fraction: 0.01
batch_size: 256
max_samples: 10000

# Hardware
device: "cuda"

# Output and visualization
plot_results: true
save_dir: "figures/diffusion_ood/hopper"
save_model_path: "diffusion/hopper_ood"
```

## Model Directory Requirements

Your diffusion model directory must have this structure:

```
/public/gormpo/models/hopper/diffusion/
├── checkpoint.pt                    # Trained model weights
├── model.pt                         # Optional: final model
├── scheduler/                       # Scheduler configuration
│   └── scheduler_config.json       # DDIM/DDPM scheduler params
└── used_config.yaml                # Optional: training config
```

### Key Files

1. **checkpoint.pt**: Contains model state dict and config
   - Must include: `model_state_dict`, `cfg`, `target_dim`
   - Created during training by `ddim_training_unconditional.py`

2. **scheduler/**: Directory with scheduler configuration
   - Created by `DDIMScheduler.save_pretrained()` or `DDPMScheduler.save_pretrained()`
   - Contains `scheduler_config.json` with diffusion parameters

## Customization

### Creating a Custom Config

To create a config for a new environment:

1. **Copy an existing config**:
   ```bash
   cp configs/diffusion/hopper_ood.yaml configs/diffusion/myenv_ood.yaml
   ```

2. **Update model and data paths**:
   ```yaml
   model_dir: "/path/to/myenv/diffusion"
   data_path: "/path/to/myenv-dataset.pkl"
   ```

3. **Adjust parameters** (if needed):
   ```yaml
   batch_size: 128  # Reduce if OOM
   max_samples: 5000  # Reduce for faster evaluation
   ```

### Common Modifications

#### For Faster Evaluation (Less Data)
```yaml
max_samples: 5000  # Reduce from 10000
batch_size: 256    # Keep default
```

#### For Lower Memory Usage
```yaml
batch_size: 128    # Reduce from 256
max_samples: 10000 # Keep default
```

#### For Stricter OOD Detection
```yaml
anomaly_fraction: 0.005  # 0.5% instead of 1%
```

#### For More Lenient OOD Detection
```yaml
anomaly_fraction: 0.05  # 5% instead of 1%
```

## Outputs

After running OOD detection, files are saved to `save_dir`:

```
figures/diffusion_ood/hopper/
├── diffusion_roc_curve.png           # ROC curve with AUC
├── diffusion_train_distribution.png  # Train/val log-likelihood distributions
└── diffusion_ood_distribution.png    # OOD vs validation comparison
```

If `save_model_path` is specified:
```
diffusion/
└── hopper_ood_metadata.pkl           # Threshold and statistics
```

## Environment-Specific Notes

### Hopper
- Observation dim: 11
- Action dim: 3
- Total dim: 14
- Typical evaluation time: ~5-10 minutes (10K samples, batch 256)

### HalfCheetah
- Observation dim: 17
- Action dim: 6
- Total dim: 23
- Typical evaluation time: ~8-15 minutes (10K samples, batch 256)

### Walker2d
- Observation dim: 17
- Action dim: 6
- Total dim: 23
- Typical evaluation time: ~8-15 minutes (10K samples, batch 256)

## Performance Comparison

| Method | Speed | Memory | Likelihood Quality |
|--------|-------|--------|-------------------|
| RealNVP | Fast | Low | Exact |
| Neural ODE | Slow | Medium | Exact |
| **Diffusion** | **Slowest** | **High** | **ELBO (lower bound)** |

### Diffusion-Specific Considerations

1. **Slower than other methods**: Requires T denoising steps (50-1000)
2. **More memory-intensive**: Stores intermediate timestep computations
3. **ELBO bound**: Provides lower bound on log-likelihood, not exact value
4. **No gradient requirement**: Unlike Neural ODE, doesn't need backprop for likelihood

## Tips for Good Results

1. **Model Quality**: Ensure diffusion model is well-trained
   - Check training NLL/ELBO convergence
   - Verify generated samples look reasonable

2. **Batch Size**: Start with 256, reduce if OOM
   - 256: Good balance for most GPUs
   - 128: Better for limited memory
   - 64: Use only if necessary (slower)

3. **Data Sampling**: Use `max_samples` to control speed
   - 10000: Default, good coverage
   - 5000: Faster, still reasonable
   - 3000: Quick testing

4. **Threshold Selection**: Adjust based on task
   - Safety-critical: Use 0.005 (0.5%) for very strict
   - Standard: Use 0.01 (1%) default
   - Exploratory: Use 0.05 (5%) for lenient

## Troubleshooting

### Issue: "Missing checkpoint.pt"
**Solution:** Check that your `model_dir` points to the correct location:
```bash
ls /public/gormpo/models/hopper/diffusion/checkpoint.pt
```

### Issue: "Missing scheduler directory"
**Solution:** Ensure the scheduler was saved during training:
```bash
ls /public/gormpo/models/hopper/diffusion/scheduler/scheduler_config.json
```

### Issue: Very slow execution
**Solution:** Reduce `max_samples` and/or `batch_size`:
```yaml
max_samples: 3000
batch_size: 128
```

### Issue: Out of memory
**Solution:** Reduce batch size significantly:
```yaml
batch_size: 64  # or even 32
```

### Issue: Low AUC scores
**Possible causes:**
1. Model not well-trained
2. OOD data too similar to training data
3. Try different `anomaly_fraction` values
4. Check that scheduler was loaded correctly

## Command-Line Overrides

You can override any config parameter from the command line:

```bash
# Override batch size
python diffusion/diffusion_ood.py \
  --config configs/diffusion/hopper_ood.yaml \
  --batch-size 128

# Override multiple parameters
python diffusion/diffusion_ood.py \
  --config configs/diffusion/hopper_ood.yaml \
  --batch-size 128 \
  --max-samples 5000 \
  --anomaly-fraction 0.05 \
  --device cuda

# Use CPU instead of GPU
python diffusion/diffusion_ood.py \
  --config configs/diffusion/hopper_ood.yaml \
  --device cpu
```

## Example Workflow

```bash
# 1. Verify model exists
ls -l /public/gormpo/models/hopper/diffusion/

# 2. Run OOD detection with default settings
python diffusion/diffusion_ood.py --config configs/diffusion/hopper_ood.yaml

# 3. If too slow, run with reduced data
python diffusion/diffusion_ood.py \
  --config configs/diffusion/hopper_ood.yaml \
  --max-samples 5000 \
  --batch-size 128

# 4. Check results
ls figures/diffusion_ood/hopper/
cat diffusion/hopper_ood_metadata.pkl
```

## Additional Resources

- **Diffusion OOD Documentation**: `diffusion/README_OOD.md`
- **Training Configs**: See your model's `used_config.yaml`
- **Inference Script**: `diffusion/monte_carlo_sampling_unconditional.py`
- **ELBO Implementation**: `diffusion/ddim_training_unconditional.py` (see `log_prob_elbo()`)

## Comparison with Neural ODE Configs

Diffusion configs are similar to Neural ODE configs but with key differences:

| Parameter | Neural ODE | Diffusion |
|-----------|------------|-----------|
| Model specification | `model_path` (single file) | `model_dir` (directory) |
| Batch size | 512 typical | 256 typical (more memory) |
| Max samples | 6000-10000 | 10000 (can reduce for speed) |
| Speed | Slow (ODE solving) | Slower (T denoising steps) |

Both use the same approach: ELBO/exact likelihood → threshold → OOD detection.
