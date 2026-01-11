import argparse
import os
import datetime
import random
import wandb
import numpy as np
import torch
import pandas as pd
# import gymnasium as gym
import d4rl
import d4rl.gym_mujoco  # Explicit import to register MuJoCo environments
import gym
from matplotlib import pyplot as plt
import yaml
from train import train
from torch.utils.tensorboard import SummaryWriter
from train import train
from helpers.evaluate_d4rl import _evaluate as evaluate_d4rl
from common.logger import Logger
from common.util import set_device_and_logger
from wrapper import (RandomNormalNoisyActions,
                                      RandomNormalNoisyTransitions,
                                        RandomNormalNoisyTransitionsActions
                                    )
from common import util
import warnings
warnings.filterwarnings("ignore")


def extract_dataset_variant(data_path):
    """
    Extract dataset variant identifier from data_path for model saving.

    Examples:
        '/path/to/halfcheetah_medium_expert_sparse_57.5.pkl' -> 'sparse_57.5'
        '/path/to/halfcheetah_medium_expert_sparse_72.5.pkl' -> 'sparse_72.5'
        '/path/to/halfcheetah_medium_expert_sparse.pkl' -> 'sparse'
        '/path/to/halfcheetah_medium_expert.pkl' -> ''
        None (D4RL default) -> ''

    Args:
        data_path: Path to dataset file or None

    Returns:
        Dataset variant string (e.g., 'sparse_57.5') or empty string for normal datasets
    """
    if data_path is None:
        return ''

    # Extract filename from path
    filename = os.path.basename(data_path)
    # Remove extension
    name_without_ext = os.path.splitext(filename)[0]

    # Look for 'sparse' keyword
    if 'sparse' in name_without_ext.lower():
        # Try to extract everything after the task name that includes 'sparse'
        # Pattern: taskname_sparse or taskname_sparse_X or taskname_sparse_X.Y
        parts = name_without_ext.split('_')

        # Find where 'sparse' appears
        try:
            sparse_idx = [i for i, p in enumerate(parts) if 'sparse' in p.lower()][0]
            # Join 'sparse' and everything after it
            variant = '_'.join(parts[sparse_idx:])
            return variant
        except IndexError:
            return 'sparse'

    return ''



def get_args():
    print("Running", __file__)
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default="configs/diffusion/halfcheetah_normal.yaml")
    config_args, remaining_argv = config_parser.parse_known_args()
    if config_args.config:
        with open(config_args.config, "r") as f:
            config = yaml.safe_load(f)
            config = {k.replace("-", "_"): v for k, v in config.items()}
    else:
        config = {}
    parser = argparse.ArgumentParser(parents=[config_parser])
    parser.add_argument("--algo-name", type=str, default="gormpo")
 
    parser.add_argument("--policy_path" , type=str, default="")
    parser.add_argument("--model_path" , type=str, default="/public/gormpo/models/rl/")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument(
                    "--devid", 
                    type=int,
                    default=5,
                    help="Which GPU device index to use"
                )

    parser.add_argument("--task", type=str, default="halfcheetah-v2")
    parser.add_argument("--seeds", type=int, nargs='+', default=[1])
    parser.add_argument("--actor-lr", type=float, default=3e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument('--auto-alpha', default=True)
    parser.add_argument('--target-entropy', type=int, default=-1) #-action_dim
    parser.add_argument('--alpha-lr', type=float, default=3e-4)

    # dynamics model's arguments
    parser.add_argument("--dynamics-lr", type=float, default=0.001)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--reward-penalty-coef", type=float, default=0.5) #1e=6
    parser.add_argument("--rollout-length", type=int, default=5) #1 
    parser.add_argument("--rollout-batch-size", type=int, default=50000) #50000
    parser.add_argument("--rollout-freq", type=int, default=1000)
    parser.add_argument("--model-retain-epochs", type=int, default=5)
    parser.add_argument("--real-ratio", type=float, default=0.05)
    parser.add_argument("--dynamics-model-dir", type=str, default=None)
    parser.add_argument("--penalty_type", type=str, default="linear", choices=["linear", "inverse", "exponential", "softplus"])

    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--step-per-epoch", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--terminal_counter", type=int, default=1)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--results_output", type=str, default=None, help="Optional: Path to shared CSV file for accumulating results across multiple runs")
    
    parser.add_argument("--density_model", type=str, default="realnvp")
    parser.add_argument("--classifier_model_name", type=str, default="neuralode")
    #============ noisy mujoco arguments ============
    parser.add_argument("--noise_rate_action", type=float, help="Portion of action to be noisy with probability", default=0.01)
    parser.add_argument("--noise_rate_transition", type=float, help="Portion of transitions to be noisy with probability", default=0.01)
    parser.add_argument("--loc", type=float, default=0.0, help="Mean of the noise distribution")
    parser.add_argument("--scale_action", type=float, default=0.001, help="Standard deviation of the action noise distribution")
    parser.add_argument("--scale_transition", type=float, default=0.001, help="Standard deviation of the transition noise distribution")
    parser.add_argument("--action", action='store_true', help="Create dataset with noisy actions")
    parser.add_argument("--transition", action='store_true', help="Create dataset with noisy transitions")
    #======================================================

    parser.add_argument(
        '--root-dir', 
        #default='log/hopper-medium-replay-v0/mopo',
         default='log', help='root dir'
    )

    parser.set_defaults(**config)

    # 5. Final parse (command line still wins over YAML)
    args = parser.parse_args(remaining_argv)
    args.config = config_args.config
    print(args.config)

    return args



def main(args):
    run = wandb.init(
                project=args.task,
                group=args.algo_name,
                config=vars(args),
                )
    results = []
    for seed in args.seeds:
        print(f"Running with {args.reward_penalty_coef} penalty coef")
        args.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if args.device != "cpu":
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


        t0 = datetime.datetime.now().strftime("%m%d_%H%M%S")

        # Extract dataset variant from data_path for model saving differentiation
        dataset_variant = extract_dataset_variant(args.data_path)
        variant_suffix = f'_{dataset_variant}' if dataset_variant else ''

        log_file = f'seed_{seed}_{t0}_{args.task.replace("-", "_")}_{args.algo_name}{variant_suffix}'
        log_path = os.path.join(args.logdir, args.algo_name, args.density_model,log_file)

        model_path = os.path.join(args.model_path, args.task.lower(), args.density_model, log_file)
        writer = SummaryWriter(log_path)
        writer.add_text("args", str(args))
        logger = Logger(writer=writer,log_path=log_path)
        model_logger = Logger(writer=writer,log_path=model_path)

        Devid = args.devid if args.device == 'cuda' else -1
        set_device_and_logger(Devid, logger, model_logger)

        args.model_path = model_path
        args.device = util.device

        env = gym.make(args.task)
        
        if args.action and not args.transition:
            print("Environment with noisy actions")
            env = RandomNormalNoisyActions(env=env, noise_rate=args.noise_rate_action, loc = args.loc, scale = args.scale_action)
        elif args.transition and not args.action:
            print("Environment with noisy transitions")
            env = RandomNormalNoisyTransitions(env=env, noise_rate=args.noise_rate_transition, loc = args.loc, scale = args.scale_transition)
        elif args.transition and args.action:
            print("Environment with noisy actions and transitions")
            env = RandomNormalNoisyTransitionsActions(env=env, noise_rate_action=args.noise_rate_action, loc = args.loc, scale_action = args.scale_action,\
                                                            noise_rate_transition=args.noise_rate_transition, scale_transition = args.scale_transition)
        else:
            print("Environment without noise")
            env = env
         
        policy, trainer = train(env, run, logger, seed, args)
        trainer.algo.save_dynamics_model(f"dynamics_model")

        eval_res = evaluate_d4rl(policy, env, args.eval_episodes, args=args, plot=True)
        eval_res['seed']= seed
        results.append(eval_res)
        


    # Save results to CSV
    if args.results_output:
        # Use specified output path for shared results across multiple runs
        results_path = args.results_output
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        # Append to existing file if it exists, otherwise create new
        results_df = pd.DataFrame(results)
        if os.path.exists(results_path):
            # Read existing results and append new ones
            existing_df = pd.read_csv(results_path)
            combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            combined_df.to_csv(results_path, index=False)
            print(f"Results appended to {results_path}")
        else:
            results_df.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")
    else:
        # Default behavior: save with timestamp
        os.makedirs(os.path.join('results',args.task.lower() + variant_suffix, args.density_model), exist_ok=True)
        results_df = pd.DataFrame(results)
        results_path = os.path.join('results',args.task.lower() + variant_suffix, args.density_model, f"results_{t0}.csv")
        results_df.to_csv(results_path, index=False)
        print(f"Results saved to {results_path}")

    wandb.finish()

if __name__ == "__main__":

  
    main(args=get_args())
