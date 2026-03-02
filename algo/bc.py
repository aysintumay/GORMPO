import argparse
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from datetime import datetime
import gym
import d4rl
import d4rl.gym_mujoco  # Explicit import to register MuJoCo environments

# add parent directory to path
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.policy_models import MLP, DiagGaussian, BasicActor
from wrapper import (
    RandomNormalNoisyActions,
    RandomNormalNoisyTransitions,
    RandomNormalNoisyTransitionsActions,
)

BC_MODEL_DIR = "/public/gormpo/models/bc"

# run command in the GORMPO directory:
# python algo/bc.py --task halfcheetah-medium-expert-v2 --data_path /public/d4rl/sparse_datasets/halfcheetah_medium_expert_sparse_72.5.pkl --seeds 1 2 3 --epochs 25 --device_id 0


def get_dataset_name(args):
    """Derive a dataset name from data_path (filename stem) or task name."""
    if args.data_path is not None:
        return os.path.splitext(os.path.basename(args.data_path))[0]
    return args.task


def get_model_path(args, seed):
    """Return canonical save path: /public/gormpo/models/bc/<env-name>/<dataset_name>_seed<N>.pkl"""
    dataset_name = get_dataset_name(args)
    return os.path.join(BC_MODEL_DIR, args.task, f"{dataset_name}_seed{seed}.pkl")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default="halfcheetah-medium-v2")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--logdir", type=str, default="results")
    parser.add_argument("--eval_episodes", type=int, default=15)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--data_path", type=str, default=None)
    # noisy environment arguments (matching mopo.py)
    parser.add_argument("--noise_rate_action", type=float, default=0.0)
    parser.add_argument("--noise_rate_transition", type=float, default=0.0)
    parser.add_argument("--loc", type=float, default=0.0)
    parser.add_argument("--scale_action", type=float, default=0.0)
    parser.add_argument("--scale_transition", type=float, default=0.0)
    parser.add_argument("--action", action='store_true', help="Use noisy actions")
    parser.add_argument("--transition", action='store_true', help="Use noisy transitions")
    return parser.parse_args()


def make_env(args):
    """Create environment, matching mopo.py's structure."""
    env = gym.make(args.task)

    if args.action and not args.transition:
        print("Environment with noisy actions")
        env = RandomNormalNoisyActions(
            env=env, noise_rate=args.noise_rate_action,
            loc=args.loc, scale=args.scale_action,
        )
    elif args.transition and not args.action:
        print("Environment with noisy transitions")
        env = RandomNormalNoisyTransitions(
            env=env, noise_rate=args.noise_rate_transition,
            loc=args.loc, scale=args.scale_transition,
        )
    elif args.transition and args.action:
        print("Environment with noisy actions and transitions")
        env = RandomNormalNoisyTransitionsActions(
            env=env,
            noise_rate_action=args.noise_rate_action, loc=args.loc,
            scale_action=args.scale_action,
            noise_rate_transition=args.noise_rate_transition,
            scale_transition=args.scale_transition,
        )
    else:
        print("Environment without noise")

    return env


class BehaviorCloning:
    def __init__(self, args, env, seed=0):
        self.args = args
        self.device = torch.device(f"cuda:{args.device_id}" if args.device == "cuda" else "cpu")

        self.env = env
        self.env.seed(seed)

        # Dataset loading matching train.py
        if args.data_path is not None:
            try:
                with open(args.data_path, "rb") as f:
                    self.dataset = pickle.load(f)
                    print('opened the pickle file for synthetic dataset')
            except:
                self.dataset = np.load(args.data_path)
                self.dataset = {k: self.dataset[k] for k in self.dataset.files}
                print('opened the npz file for synthetic dataset')
        else:
            self.dataset = d4rl.qlearning_dataset(self.env)

        args.obs_shape = self.env.observation_space.shape
        args.action_dim = np.prod(self.env.action_space.shape)

        # Initialize model
        self.obs_dim = self.env.observation_space.shape[0]
        self.action_dim = args.action_dim

        actor_backbone = MLP(input_dim=self.obs_dim, hidden_dims=[256, 256, 256])
        dist = DiagGaussian(
            latent_dim=getattr(actor_backbone, "output_dim"),
            output_dim=self.action_dim,
            unbounded=True,
            conditioned_sigma=True,
        )
        self.model = BasicActor(actor_backbone, dist, self.env.action_space, self.device)
        self.optimizer = optim.Adam(self.model.actor.parameters(), lr=args.lr)

        # Create data tensors
        self.obs = torch.FloatTensor(self.dataset["observations"]).to(self.device)
        self.actions = torch.FloatTensor(self.dataset["actions"]).to(self.device)

    def train(self):
        n_samples = len(self.obs)
        print(f"Training on {n_samples} samples")
        n_batches = n_samples // self.args.batch_size
        prev_loss = np.inf
        mse = nn.MSELoss()

        for epoch in range(self.args.epochs):
            total_loss = 0
            indices = np.random.permutation(n_samples)

            for i in range(n_batches):
                batch_idx = indices[i * self.args.batch_size:(i + 1) * self.args.batch_size]
                obs_batch = self.obs[batch_idx]
                actions_batch = self.actions[batch_idx]

                self.optimizer.zero_grad()
                action, _ = self.model(obs_batch)
                loss = mse(action, actions_batch)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / n_batches
            print(f"Epoch {epoch + 1}/{self.args.epochs}, Loss: {avg_loss:.4f}")
            if i > 0 and abs(avg_loss - prev_loss) < 0.0005:
                print(f"Early stopping at epoch {epoch + 1}")
                break
            prev_loss = avg_loss

    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model.state_dict(), f)
        print(f"Model saved to {path}")

    def load_model(self, path):
        with open(path, "rb") as f:
            state_dict = pickle.load(f)
        self.model.load_state_dict(state_dict)
        print(f"Model loaded from {path}")


def _evaluate(policy, eval_env, episodes, args=None, plot=None):  # noqa: ARG001
    """Evaluate a policy, matching helpers/evaluate_d4rl.py _evaluate signature."""
    policy.eval()
    obs = eval_env.reset()
    eval_ep_info_buffer = []
    num_episodes = 0
    episode_reward, episode_length = 0, 0

    while num_episodes < episodes:
        action = policy.sample_action(obs, deterministic=True)
        next_obs, reward, terminal, _ = eval_env.step(action)
        episode_reward += reward
        episode_length += 1
        obs = next_obs

        if terminal:
            eval_ep_info_buffer.append(
                {"episode_reward": episode_reward, "episode_length": episode_length}
            )
            num_episodes += 1
            episode_reward, episode_length = 0, 0
            obs = eval_env.reset()

    ep_reward_mean = np.mean([ep["episode_reward"] for ep in eval_ep_info_buffer])
    ep_reward_std  = np.std( [ep["episode_reward"] for ep in eval_ep_info_buffer])
    ep_length_mean = np.mean([ep["episode_length"] for ep in eval_ep_info_buffer])
    ep_length_std  = np.std( [ep["episode_length"] for ep in eval_ep_info_buffer])

    return {
        "mean_return": ep_reward_mean,
        "std_return":  ep_reward_std,
        "mean_length": ep_length_mean,
        "std_length":  ep_length_std,
    }


def main():
    args = get_args()

    t0 = datetime.now().strftime("%m%d_%H%M%S")

    if args.device_id < 0 or not torch.cuda.is_available():
        args.device = "cpu"
    print(f"Using device: {args.device}" + (f":{args.device_id}" if args.device == "cuda" else ""))

    results = []
    for seed in args.seeds:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        env = make_env(args)

        bc = BehaviorCloning(args, env, seed=seed)
        bc.train()

        # Save model
        model_path = get_model_path(args, seed)
        bc.save_model(model_path)

        # Evaluate
        eval_res = _evaluate(bc.model, env, args.eval_episodes, args=args)
        eval_res['seed'] = seed
        results.append(eval_res)

        print(f"Seed {seed} - Mean Return: {eval_res['mean_return']:.2f} ± {eval_res['std_return']:.2f}")

    # Save results to CSV
    os.makedirs(os.path.join(args.logdir, args.task, "bc"), exist_ok=True)
    results_df = pd.DataFrame(results)
    results_path = os.path.join(args.logdir, args.task, "bc", f"bc_results_{t0}.csv")
    results_df.to_csv(results_path, index=False)
    print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
