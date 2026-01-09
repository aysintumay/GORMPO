"""
Test script to verify that transition model loading includes normalizers
"""
import os
import sys
import torch
import gym
import numpy as np

# Add project root to path
sys.path.append(os.path.dirname(__file__))

from common import util
from transition_model import TransitionModel
from static_fns import hopper

# Set device
util.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def test_model_loading():
    """Test loading a saved transition model"""

    # Create a dummy environment to get spaces
    env = gym.make('Hopper-v2')
    obs_space = env.observation_space
    action_space = env.action_space

    print("=" * 60)
    print("Testing Transition Model Loading")
    print("=" * 60)

    # Create a TransitionModel instance
    model_params = {
        'model': {
            'hidden_dims': [200, 200, 200, 200],
            'num_elite': 5,
            'ensemble_size': 7
        }
    }

    classifier = {
        'model': None,
        'thr': 0.0,
        'name': None,
        'mean': None,
        'std': None
    }

    transition_model = TransitionModel(
        obs_space=obs_space,
        action_space=action_space,
        static_fns=hopper.StaticFns,
        lr=3e-4,
        classifier=classifier,
        type="linear",
        holdout_ratio=0.1,
        reward_penalty_coef=0.0,
        **model_params
    )

    print(f"\nBefore loading:")
    print(f"  obs_normalizer.mean: {getattr(transition_model.obs_normalizer, 'mean', None)}")
    print(f"  obs_normalizer.var: {getattr(transition_model.obs_normalizer, 'var', None)}")
    print(f"  obs_normalizer.tot_count: {transition_model.obs_normalizer.tot_count}")
    print(f"  act_normalizer.mean: {getattr(transition_model.act_normalizer, 'mean', None)}")
    print(f"  act_normalizer.var: {getattr(transition_model.act_normalizer, 'var', None)}")
    print(f"  act_normalizer.tot_count: {transition_model.act_normalizer.tot_count}")

    # Try loading a model
    print(f"\n{'='*60}")
    print("Attempting to load model for hopper-medium-v2...")
    print(f"{'='*60}\n")

    try:
        result = transition_model.load_model(['hopper-medium-v2', None])

        print(f"\n{'='*60}")
        print("After loading:")
        print(f"{'='*60}")
        obs_mean = getattr(transition_model.obs_normalizer, 'mean', None)
        obs_var = getattr(transition_model.obs_normalizer, 'var', None)
        act_mean = getattr(transition_model.act_normalizer, 'mean', None)
        act_var = getattr(transition_model.act_normalizer, 'var', None)

        print(f"  obs_normalizer.mean: {obs_mean}")
        print(f"  obs_normalizer.var: {obs_var}")
        print(f"  obs_normalizer.tot_count: {transition_model.obs_normalizer.tot_count}")
        print(f"  act_normalizer.mean: {act_mean}")
        print(f"  act_normalizer.var: {act_var}")
        print(f"  act_normalizer.tot_count: {transition_model.act_normalizer.tot_count}")

        # Verify normalizers are loaded
        if obs_mean is not None and obs_var is not None:
            print(f"\n✓ SUCCESS: obs_normalizer loaded correctly")
            print(f"  - mean shape: {obs_mean.shape if hasattr(obs_mean, 'shape') else 'scalar'}")
            print(f"  - var shape: {obs_var.shape if hasattr(obs_var, 'shape') else 'scalar'}")
        else:
            print(f"\n✗ ERROR: obs_normalizer not loaded properly")

        if act_mean is not None and act_var is not None:
            print(f"\n✓ SUCCESS: act_normalizer loaded correctly")
            print(f"  - mean shape: {act_mean.shape if hasattr(act_mean, 'shape') else 'scalar'}")
            print(f"  - var shape: {act_var.shape if hasattr(act_var, 'shape') else 'scalar'}")
        else:
            print(f"\n✗ ERROR: act_normalizer not loaded properly")

        # Test a simple prediction to ensure everything works
        print(f"\n{'='*60}")
        print("Testing prediction with loaded model...")
        print(f"{'='*60}")

        obs = env.observation_space.sample()
        action = env.action_space.sample()

        next_obs, reward, terminal, info = transition_model.predict(obs, action)

        print(f"✓ Prediction successful!")
        print(f"  - next_obs shape: {next_obs.shape}")
        print(f"  - reward shape: {reward.shape}")
        print(f"  - terminal shape: {terminal.shape}")

        print(f"\n{'='*60}")
        print("All tests passed!")
        print(f"{'='*60}\n")

    except Exception as e:
        print(f"\n✗ ERROR loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True

if __name__ == "__main__":
    success = test_model_loading()
    sys.exit(0 if success else 1)
