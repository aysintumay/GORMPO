# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**GORMPO (Generative OOD-regularized Model-based Policy Optimization)** is a model-based offline reinforcement learning framework that uses density estimation (RealNVP normalizing flows) to detect out-of-distribution (OOD) states and penalize uncertainty in world model predictions. It extends MOPO (Model-based Offline Policy Optimization) with KDE-based regularization.

## Key Commands

### Training

Run GORMPO training with YAML config:
```bash
python mopo.py --config configs/hopper_linear.yaml
```

Run with command-line arguments:
```bash
python mopo.py --task halfcheetah-medium-v2 --algo-name gormpo --epoch 100 --reward-penalty-coef 0.5 --penalty_type linear
```

Run with multiple seeds:
```bash
python mopo.py --config configs/walker2d_linear_normal.yaml --seeds 1 2 3 4 5
```

### Testing

Test RealNVP density model:
```bash
cd realnvp_module
python realnvp.py  # Built-in tests
```

Test dataset loading utilities:
```bash
cd realnvp_module
python test_dataset_loader.py
python example_usage.py
```

### Environment Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Required external setup:
- D4RL datasets (automatically downloaded via `d4rl` package)
- MuJoCo physics engine
- Pre-trained RealNVP classifiers (specified via `--classifier_model_name`)

## Architecture

### Core Training Flow

1. **Entry Point**: `mopo.py` parses arguments and initializes WandB tracking
2. **Training Pipeline**: `train.py` sets up:
   - SAC policy (actor-critic)
   - TransitionModel (ensemble dynamics with KDE penalty)
   - Offline and model buffers
   - MOPO algorithm orchestrator
3. **Trainer Execution**: `trainer.py` executes:
   - `train_dynamics()`: Pre-trains world model on offline dataset
   - `train_policy()`: Trains SAC policy using model rollouts

### Key Components

**TransitionModel** (`transition_model.py`):
- Manages ensemble of dynamics models (`EnsembleModel`)
- Implements KDE-based OOD penalty via `_return_kde_penalty()`
- Penalty types: `linear`, `inverse`, `tanh`, `softplus`
- Normalizes observations/actions via `StandardNormalizer`

**MOPO Algorithm** (`algo/mopo.py`):
- `learn_dynamics()`: Trains ensemble with hold-out validation
- `rollout_transitions()`: Generates synthetic trajectories from policy
- Manages offline buffer (real data) and model buffer (synthetic data)

**SAC Policy** (`algo/sac.py`):
- Standard Soft Actor-Critic with automatic entropy tuning
- Twin Q-networks with target networks
- Used for both model rollouts and policy learning

**RealNVP Classifier** (`realnvp_module/realnvp.py`):
- Normalizing flow for density estimation
- Returns log-probability scores for state-action pairs
- Threshold-based anomaly detection for OOD identification

**Static Functions** (`static_fns/`):
- Environment-specific termination functions
- Separate files for `hopper`, `halfcheetah`, `walker2d`, `abiomed`
- Define episode termination logic for model rollouts

### Configuration System

Configurations are split across Python dicts and YAML files:

**Python configs** (`configs/*.py`):
- Define `default_config` with `transition_params` and `mopo_params`
- Environment-specific hyperparameters (ensemble size, hidden dims, etc.)

**YAML configs** (`configs/*.yaml`):
- Override specific parameters for experiments
- Specify task, dataset path, penalty type, training epochs
- YAML values override Python defaults, CLI args override both

Example config structure:
```python
default_config = {
    "transition_params": {
        "model": {
            "hidden_dims": [200, 200, 200, 200],
            "num_elite": 5,
            "ensemble_size": 7
        }
    },
    "mopo_params": {
        "max_epoch": 400,
        "rollout_batch_size": 50000
    }
}
```

### Data Flow

1. **Dataset Loading**:
   - D4RL datasets: `d4rl.qlearning_dataset(env)`
   - Custom datasets: Pickle or NPZ via `--data_path`
   - Loaded into `ReplayBuffer` as offline buffer

2. **World Model Training**:
   - Ensemble trained on offline data with hold-out validation
   - Normalizers fit on training split
   - Elite models selected based on validation MSE

3. **Policy Training Loop**:
   - Sample initial states from offline buffer
   - Roll out trajectories using current policy + world model
   - KDE penalty applied to rewards during rollouts
   - SAC trained on mixture of real (5%) and model data (95%)

4. **KDE Penalty Computation**:
   - Concatenate state-action or action-reward pairs
   - Compute log-probability via pre-trained RealNVP
   - Apply penalty function (linear/inverse/tanh/softplus)
   - Penalize rewards for OOD samples during rollouts

### Noisy Environment Wrappers

`wrapper.py` provides gym wrappers for noise injection:
- `RandomNormalNoisyActions`: Adds Gaussian noise to actions
- `RandomNormalNoisyTransitions`: Adds noise to observations
- `RandomNormalNoisyTransitionsActions`: Combined noise

Used for robustness testing with `--action` and `--transition` flags.

## Development Notes

### File Organization
- `algo/`: SAC, MOPO, BCQ algorithm implementations
- `models/`: Neural network architectures (policy, ensemble dynamics)
- `common/`: Shared utilities (buffer, logger, normalizer, device management)
- `helpers/`: Evaluation, plotting, scoring utilities
- `static_fns/`: Environment-specific termination functions
- `realnvp_module/`: Density estimation model and dataset utilities
- `configs/`: Hyperparameter configurations
- `results/`: Saved evaluation results (CSV)
- `log/`: TensorBoard logs
- `wandb/`: Weights & Biases experiment tracking

### Critical Implementation Details

**Ensemble Dynamics**:
- Probabilistic model outputting mean and log-variance
- Max/min logvar bounds with learnable parameters
- Elite model selection based on validation performance
- Predictions sampled from elite models during rollouts

**KDE Integration**:
- Pre-trained RealNVP must be provided via `--classifier_model_name`
- Classifier expects normalized inputs (mean/std stored during training)
- Threshold determined during RealNVP training
- Penalty applied per-step during model rollouts, not during world model training

**Buffer Management**:
- Offline buffer: Fixed size, loaded from dataset
- Model buffer: Rolling window, retains data for `model_retain_epochs`
- Sampling: Mix real and synthetic data based on `real_ratio` (default 0.05)

**Training Sequence**:
1. World model trained to convergence on offline data
2. Policy training begins with periodic model rollouts
3. Rollouts happen every `rollout_freq` steps (default 1000)
4. Model buffer populated with synthetic trajectories
5. SAC updates using mixed real/synthetic batches

### Important Parameters

**Penalty Configuration**:
- `reward-penalty-coef`: Scale of OOD penalty (0.0-1.0, typical: 0.001-1.0)
- `penalty_type`: Function shape (linear, inverse, tanh, softplus)
- `classifier_model_name`: Path to pre-trained RealNVP model

**Model Parameters**:
- `n-ensembles`: Total models in ensemble (default: 7)
- `n-elites`: Models used for predictions (default: 5)
- `rollout-length`: Steps per synthetic trajectory (default: 5)
- `rollout-batch-size`: Initial states for rollouts (default: 50000)

**Training Parameters**:
- `epoch`: Training epochs (default: 100)
- `step-per-epoch`: SAC updates per epoch (default: 1000)
- `real-ratio`: Fraction of real data in training batch (default: 0.05)
- `batch-size`: SAC training batch size (default: 256)

### Common Tasks

**Adding a new environment**:
1. Create `static_fns/{env_name}.py` with `StaticFns.termination_fn()`
2. Add config in `configs/{env_name}.py` with `default_config`
3. Train RealNVP classifier on environment's offline dataset
4. Update task name in `--task` argument

**Modifying penalty function**:
- Edit `TransitionModel._return_kde_penalty()` in `transition_model.py`
- Add new type to `--penalty_type` choices in `mopo.py`
- Penalty is computed during `dynamics_model.predict()` calls in rollouts

**Debugging model rollouts**:
- Check `transition_model.py:predict()` for reward modification
- Verify classifier loading in `train.py` (lines 113-116)
- Monitor model buffer size in trainer logs
- Use TensorBoard to track rollout statistics

**Evaluating trained policies**:
- Policies auto-evaluated via `helpers/evaluate_d4rl.py` after training
- Results saved to `results/{task}/{density_model}/results_{timestamp}.csv`
- Evaluation runs `eval_episodes` (default: 100) rollouts in real environment
- Normalized scores computed using D4RL benchmark standards

### WandB Integration

Training metrics logged to Weights & Biases:
- Project name: `{task}` (e.g., "Hopper", "halfcheetah-medium-v2")
- Group: `{algo-name}` (e.g., "gormpo")
- Automatic logging of: policy loss, Q-values, model MSE, rollout statistics
- Plots: accuracy curves, loss curves, Q-value evolution

### Device Management

GPU selection via `--devid` flag (default: 0). Device set globally in `common.util.device` after initialization. All models moved to device automatically during construction.
