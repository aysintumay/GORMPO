"""
Test script to verify saving and loading of normalizers works correctly
"""
import os
import sys
import torch
import gym
import numpy as np
import tempfile
import shutil

sys.path.append(os.path.dirname(__file__))

from common import util
from transition_model import TransitionModel
from static_fns import hopper

# Set device
util.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Mock logger for save
class MockLogger:
    def __init__(self, temp_dir):
        self.log_path = temp_dir

util.logger_model = MockLogger(tempfile.mkdtemp())

def test_save_and_load():
    """Test saving and loading normalizers"""

    # Create a dummy environment
    env = gym.make('Hopper-v2')
    obs_space = env.observation_space
    action_space = env.action_space

    print("=" * 60)
    print("Testing Save and Load of Normalizers")
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

    # Create first model and fit normalizers with dummy data
    print("\n1. Creating model and fitting normalizers with dummy data...")
    model1 = TransitionModel(
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

    # Create dummy data and fit normalizers
    obs_dim = obs_space.shape[0]
    action_dim = action_space.shape[0]
    dummy_obs = np.random.randn(1000, obs_dim).astype(np.float32)
    dummy_actions = np.random.randn(1000, action_dim).astype(np.float32)

    model1.obs_normalizer.fit(dummy_obs)
    model1.act_normalizer.fit(dummy_actions)

    print(f"   obs_normalizer fitted:")
    print(f"     - mean shape: {model1.obs_normalizer.mean.shape}")
    print(f"     - var shape: {model1.obs_normalizer.var.shape}")
    print(f"     - tot_count: {model1.obs_normalizer.tot_count}")
    print(f"   act_normalizer fitted:")
    print(f"     - mean shape: {model1.act_normalizer.mean.shape}")
    print(f"     - var shape: {model1.act_normalizer.var.shape}")
    print(f"     - tot_count: {model1.act_normalizer.tot_count}")

    # Save the model
    print(f"\n2. Saving model to: {util.logger_model.log_path}")
    model1.save_model('test_dynamics_model')

    # Check if normalizer file was created
    normalizer_path = os.path.join(util.logger_model.log_path, 'test_dynamics_model', 'normalizers.pt')
    if os.path.exists(normalizer_path):
        print(f"   ✓ normalizers.pt file created successfully")
    else:
        print(f"   ✗ ERROR: normalizers.pt file not found")
        return False

    # Store original values for comparison
    orig_obs_mean = model1.obs_normalizer.mean.copy() if isinstance(model1.obs_normalizer.mean, np.ndarray) else model1.obs_normalizer.mean.cpu().numpy()
    orig_obs_var = model1.obs_normalizer.var.copy() if isinstance(model1.obs_normalizer.var, np.ndarray) else model1.obs_normalizer.var.cpu().numpy()
    orig_act_mean = model1.act_normalizer.mean.copy() if isinstance(model1.act_normalizer.mean, np.ndarray) else model1.act_normalizer.mean.cpu().numpy()
    orig_act_var = model1.act_normalizer.var.copy() if isinstance(model1.act_normalizer.var, np.ndarray) else model1.act_normalizer.var.cpu().numpy()

    # Create a second model and manually load from the saved directory
    print(f"\n3. Creating new model and loading saved normalizers...")
    model2 = TransitionModel(
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

    # Manually load the model (simulate load_model behavior)
    model_path = os.path.join(util.logger_model.log_path, 'test_dynamics_model', 'model.pt')
    normalizer_path = os.path.join(util.logger_model.log_path, 'test_dynamics_model', 'normalizers.pt')

    # Load model weights
    model_state = torch.load(model_path, map_location=util.device)
    model2.model.load_state_dict(model_state)

    # Load normalizers
    if os.path.exists(normalizer_path):
        normalizer_data = torch.load(normalizer_path, map_location=util.device)

        model2.obs_normalizer.mean = normalizer_data['obs_normalizer']['mean']
        model2.obs_normalizer.var = normalizer_data['obs_normalizer']['var']
        model2.obs_normalizer.tot_count = normalizer_data['obs_normalizer']['tot_count']

        model2.act_normalizer.mean = normalizer_data['act_normalizer']['mean']
        model2.act_normalizer.var = normalizer_data['act_normalizer']['var']
        model2.act_normalizer.tot_count = normalizer_data['act_normalizer']['tot_count']

        print(f"   ✓ Normalizers loaded successfully")
    else:
        print(f"   ✗ ERROR: Normalizer file not found")
        return False

    # Verify loaded values match original values
    print(f"\n4. Verifying loaded values match original...")

    loaded_obs_mean = model2.obs_normalizer.mean.cpu().numpy() if torch.is_tensor(model2.obs_normalizer.mean) else model2.obs_normalizer.mean
    loaded_obs_var = model2.obs_normalizer.var.cpu().numpy() if torch.is_tensor(model2.obs_normalizer.var) else model2.obs_normalizer.var
    loaded_act_mean = model2.act_normalizer.mean.cpu().numpy() if torch.is_tensor(model2.act_normalizer.mean) else model2.act_normalizer.mean
    loaded_act_var = model2.act_normalizer.var.cpu().numpy() if torch.is_tensor(model2.act_normalizer.var) else model2.act_normalizer.var

    obs_mean_match = np.allclose(orig_obs_mean, loaded_obs_mean)
    obs_var_match = np.allclose(orig_obs_var, loaded_obs_var)
    act_mean_match = np.allclose(orig_act_mean, loaded_act_mean)
    act_var_match = np.allclose(orig_act_var, loaded_act_var)

    print(f"   obs_normalizer.mean match: {'✓' if obs_mean_match else '✗'}")
    print(f"   obs_normalizer.var match: {'✓' if obs_var_match else '✗'}")
    print(f"   act_normalizer.mean match: {'✓' if act_mean_match else '✗'}")
    print(f"   act_normalizer.var match: {'✓' if act_var_match else '✗'}")

    # Test that predictions work
    print(f"\n5. Testing prediction with loaded model...")
    try:
        obs = env.observation_space.sample()
        action = env.action_space.sample()
        next_obs, reward, terminal, info = model2.predict(obs, action)
        print(f"   ✓ Prediction successful!")
    except Exception as e:
        print(f"   ✗ ERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Cleanup
    shutil.rmtree(util.logger_model.log_path)

    if obs_mean_match and obs_var_match and act_mean_match and act_var_match:
        print(f"\n{'='*60}")
        print("✓ ALL TESTS PASSED!")
        print(f"{'='*60}\n")
        return True
    else:
        print(f"\n{'='*60}")
        print("✗ SOME TESTS FAILED")
        print(f"{'='*60}\n")
        return False

if __name__ == "__main__":
    success = test_save_and_load()
    sys.exit(0 if success else 1)
