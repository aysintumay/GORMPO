# GORMPO

This directory contains the GORMPO project, including density estimation models, dataset utilities, and configuration files.

## Quick GORMPO training with trained density estimators

KDE

```
python mopo.py --config configs/kde/gormpo_halfcheetah_medium_expert_sparse_3.yaml --devid 0
```

VAE:

```
python mopo.py --config configs/vae/gormpo_halfcheetah_medium_expert_sparse_3.yaml --devid 0
```

RealNVP:

```
python mopo.py --config configs/realnvp/gormpo_halfcheetah_medium_expert_sparse_3.yaml --devid 0
```

Diffusion: 
```
python mopo.py --config configs/diffusion/halfcheetah_medium_expert_sparse_3.yaml --devid 0
```

NeuralODE
```
python mopo.py --config configs/neuralODE/halfcheetah_medium_expert_sparse_3.yaml --devid 0
```

Configs *must* include classifier_model_name, task, data_path (for sparse saved datasets), algo-name, reward-penalty-coef, penalty_type parameters.


For hyperparameter search, create a bash script like 'bash_scr/halfcheetah_medium_expert_v2_gormpo_kde_sparse3.sh' by specifying the *config*. Then run as follows:

```
chmod +x /bash_scr/halfcheetah_medium_expert_v2_gormpo_kde_sparse3.sh 
./bash_scr/halfcheetah_medium_expert_v2_gormpo_kde_sparse3.sh
```
## Multi-Seed Training

Run training across multiple seeds using bash scripts in `bash_scr/mult_seed/`:

```bash
bash bash_scr/mult_seed/gormpo_kde_halfcheetah_medium_expert_sparse3.sh
```

Features:
- Trains density estimator and GORMPO policy for each seed (42, 123, 456)
- Saves all results to shared CSV file
- Automatically computes normalized scores via `helpers/normalizer.py`

## OOD Testing

Evaluate density estimators on OOD datasets at different distance levels:

```bash
python test_kde_ood_levels.py \
    --model_path /public/gormpo/models/halfcheetah_medium_expert_sparse_3/kde \
    --dataset_name halfcheetah-medium-expert-v2 \
    --sparse_dataset_name halfcheetah_medium_expert_sparse_72.5 \
    --distances 1 2 3 4 \
    --device cuda
```

**Outputs** (saved to `figures/{model}_ood_distance_tests/{dataset_name}/`):
- `ood_distance_summary.png`: Log-likelihood, ROC AUC, and accuracy vs distance
- `roc_curves.png`: ROC curves for each distance level
- `log_likelihood_distributions.png`: ID vs OOD distributions
- `results.json`: Numerical results for all metrics

Available scripts: `test_kde_ood_levels.py`, `test_vae_ood_levels.py`, `test_realnvp_ood_levels.py`, `test_diffusion_ood_levels.py`, `test_neuralode_ood_levels.py`

## Results Merging

Combine OOD test results across density estimators using `notebooks/merge_results.ipynb`:
- Reads `results.json` from multiple methods (VAE, KDE, RealNVP)
- Generates comparison plots for each dataset
- Saves merged plots to `figures/merged_results/{dataset_name}_comparison.png`

## Configuration File Reference

Configs are organized by density estimator type under `configs/<density_model>/`. Each subdirectory contains three kinds of files:

### 1. Density Estimator Training Configs

Used to train the standalone density model (KDE, VAE, RealNVP, Diffusion, or Neural ODE). These are fed to the model-specific training scripts, not to `mopo.py`.


- Standard D4RL dataset: `configs/<model>/gormpo_<dataset_name>.yaml`
  - Example: `configs/kde/halfcheetah_medium_expert.yaml`
- Sparse D4RL dataset: `configs/<model>/gormpo_<dataset_name>_sparse_3.yaml`
  - Example: `configs/kde/halfcheetah_medium_expert_sparse_3.yaml`
- Training density models:  
    - NeuralODE: `configs/<model>/<dataset_name>_train.yaml`
    - KDE: `configs/<model>/<dataset_name>_train.yaml`
    - Other models: `configs/<model>/<dataset_name>.yaml`

**Common variables (all density models):**
| Variable | Description |
|---|---|
| `task` | D4RL task name (e.g., `halfcheetah-medium-expert-v2`) |
| `data_path` | Path to sparse dataset `.pkl` file; `null` to use D4RL default |
| `epochs` | Number of training epochs |
| `batch_size` / `batch` | Mini-batch size |
| `lr` | Learning rate |
| `val_ratio` / `val_size` | Fraction of data used for validation |
| `test_ratio` / `test_size` | Fraction of data used for test |
| `model_save_path` / `out` | Directory where the trained density model is saved |
| `fig_save_path` | Directory for output plots |
| `seed` | Random seed |
| `device` | Compute device (`cuda` or `cpu`) |

**KDE-specific variables** (`configs/kde/*.yaml`):
| Variable | Description |
|---|---|
| `bandwidth` | Kernel bandwidth for KDE |
| `k_neighbors` | Number of neighbors for density estimation |
| `percentile` | Percentile threshold for OOD classification |
| `pca` | Whether to apply PCA before KDE |
| `optimize_percentile` | Whether to search for the best percentile threshold |
| `optimization_metric` | Metric used when optimizing threshold (e.g., `density_range`) |
| `temporal_split` | Whether to split data temporally rather than randomly |

**VAE-specific variables** (`configs/vae/*.yaml`):
| Variable | Description |
|---|---|
| `input_dim` | Dimension of state+action input (e.g., 23 = 17 obs + 6 act for HalfCheetah) |
| `latent_dim` | Dimension of the latent space |
| `hidden_dims` | Layer widths of encoder/decoder MLP |
| `beta` | Weight of the KL term in the ELBO (`beta`-VAE coefficient) |
| `anomaly_fraction` | Fraction of validation data labelled anomalous when setting threshold |
| `patience` | Early-stopping patience in epochs |

**RealNVP-specific variables** (`configs/realnvp/*.yaml`):
| Variable | Description |
|---|---|
| `obs_dim` / `action_dim` | Observation and action dimensions |
| `num_layers` | Number of coupling layers |
| `hidden_dims` | Hidden layer widths per coupling layer |
| `anomaly_fraction` | Fraction used as anomaly threshold during evaluation |
| `patience` | Early-stopping patience in epochs |
| `anomaly_type` | Type of synthetic anomaly: `outlier` or `uniform` (used when `use_rl_data: false`) |

**Neural ODE-specific variables** (`configs/neuralODE/*.yaml`):
| Variable | Description |
|---|---|
| `hidden_dims` | Hidden layer widths of the ODE function network |
| `activation` | Activation function (`silu` or `tanh`) |
| `time_dependent` | Whether the ODE function receives time as input |
| `solver` | ODE integrator (`dopri5`, `rk4`, or `euler`) |
| `t0` / `t1` | Integration interval start/end |
| `rtol` / `atol` | Relative and absolute ODE solver tolerances |
| `wd` | Weight decay |
| `log_every` | Log training stats every N steps |
| `checkpoint_every` | Save model checkpoint every N epochs |

---

### 2. GORMPO Policy Training Configs

Used with `mopo.py` to train the GORMPO policy on top of a pre-trained density model.

**Naming convention:**
- Standard D4RL dataset: `configs/<model>/gormpo_<dataset_name>.yaml`
  - Example: `configs/kde/gormpo_halfcheetah_medium_expert.yaml`
- Sparse D4RL dataset: `configs/<model>/gormpo_<dataset_name>_sparse_3.yaml`
  - Example: `configs/kde/gormpo_halfcheetah_medium_expert_sparse_3.yaml`

The `_sparse_3` suffix denotes that the offline dataset is a sparsified version (72.5% reward sparsification) of the standard D4RL dataset.

**Variables:**
| Variable | Description |
|---|---|
| `task` | D4RL task name (e.g., `halfcheetah-medium-expert-v2`) |
| `algo-name` | Algorithm identifier; always `gormpo` for GORMPO runs |
| `data_path` | Path to sparse `.pkl` dataset; omit for standard D4RL |
| `epoch` | Number of policy training epochs |
| `rollout-length` | Steps per synthetic model rollout trajectory |
| `reward-penalty-coef` | Scale of the OOD density penalty applied to rewards |
| `penalty_type` | Shape of the penalty function: `linear`, `inverse`, `tanh`, or `softplus` |
| `classifier_model_name` | Path to the pre-trained density model directory |
| `density_model` | Density estimator type: `kde`, `vae`, `realnvp`, `diffusion`, or `neuralODE` |
| `target_dim` | Input dimensionality of the density model (obs_dim + action_dim) |
| `seed` | Random seed |

---

### 3. OOD Level Test Configs

Used with `test_<model>_ood_levels.py` scripts to evaluate a trained density model's ability to detect OOD samples at varying Mahalanobis distances.

Located under `configs/<model>/ood_test/`. The `_next` suffix variant uses a model trained on (state, next\_state, action) inputs instead of (state, action).

**Variables:**
| Variable | Description |
|---|---|
| `model_path` | Path to the trained density model |
| `dataset_name` | D4RL task name used as the in-distribution reference |
| `sparse_dataset_name` | Name of the sparse variant of the dataset |
| `base_path` | Root directory containing OOD test data files |
| `distances` | List of OOD distances to evaluate (e.g., `[0.4, 0.6, 0.8, 1]`) |
| `suffix` | OOD data file suffix (e.g., `-uniform`) |
| `device` | Compute device |
| `save_dir` | Output directory for plots and result JSON |

---

Training (`mopo.py`, `train.py`)

Enhanced model-based offline RL with KDE-based uncertainty estimation:


**Configuration Options:**
- `reward-penalty-coef`: Penalty coefficient for model uncertainty
- `penalty_type`: Type of penalty function ("linear", "inverse", "exponential", "softplus")
- `classifier_model_name`: Pre-trained KDE classifier for uncertainty estimation


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

