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

