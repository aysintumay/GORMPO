import datetime
import os
import random
import importlib
import sys
import pickle

import gym
import json
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from transition_model import TransitionModel
from models.policy_models import MLP, ActorProb, Critic, DiagGaussian
from algo.sac import SACPolicy
from algo.mopo import MOPO
from common.buffer import ReplayBuffer
from common.logger import Logger
from trainer import Trainer
from common.util import set_device_and_logger
from common import util
from realnvp_module.realnvp import RealNVP 
from vae_module.vae import VAE
from kde_module.kde import PercentileThresholdKDE
from neuralODE.neural_ode_density import ContinuousNormalizingFlow, ODEFunc
from neuralODE.neural_ode_ood import NeuralODEOOD
from diffusion.monte_carlo_sampling_unconditional import build_model_from_ckpt
from diffusion.ddim_training_unconditional import log_prob_elbo
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import d4rl
from typing import Tuple
import torch
from torch import nn
from diffusion.ddim_training_unconditional import (
    UnconditionalEpsilonMLP,
    UnconditionalEpsilonTransformer,
)


class DiffusionDensityWrapper:
    """Wrapper for diffusion model to provide score_samples interface."""

    def __init__(self, model, scheduler, target_dim, device):
        self.model = model
        self.scheduler = scheduler
        self.target_dim = target_dim
        self.device = device

    @torch.no_grad()
    def score_samples(self, x, device=None):
        """
        Compute log probability using ELBO from unconditional diffusion model.

        Args:
            x: Input samples (numpy array or tensor) of shape (batch_size, target_dim)
            device: Device to use (optional)

        Returns:
            Log probabilities normalized by target dimension
        """
        if device is None:
            device = self.device

        # Convert to tensor if needed
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()

        x = x.to(device)

        # Compute log probability using ELBO
        log_probs = log_prob_elbo(
            model=self.model,
            scheduler=self.scheduler,
            x0=x,
            device=device,
        )

        # Normalize by target dimension for consistency
        log_probs_per_dim = log_probs

        return log_probs_per_dim


def train(env, run, logger, seed, args):
    print('rollout_batch_size:', args.rollout_batch_size)
    
    if args.data_path != None:
        try:
            with open(args.data_path, "rb") as f:
                dataset = pickle.load(f)
                print('opened the pickle file for synthetic dataset')
        except:
            dataset = np.load(args.data_path)
            dataset = {k: dataset[k] for k in dataset.files}
            print('opened the npz file for synthetic dataset')
        # dataset = {k: v[:100] for k, v in dataset.items()}
        buffer_len = len(dataset['observations'])
    else:
        if args.task == "abiomed":
            dataset1 = env.world_model.data_train
            dataset2 = env.world_model.data_val
            dataset3 = env.world_model.data_test
            dataset = [dataset1, dataset2, dataset3]
            buffer_len  = len(dataset1.data) + len(dataset2.data) + len(dataset3.data)
            # dataset = dataset1
            # buffer_len  = len(dataset.data) 
            # dataset.data = dataset.data[:5]
            # dataset.pl = dataset.pl[:5]
            # dataset.labels = dataset.labels[:5]

        else:
            dataset = d4rl.qlearning_dataset(env) 
    

    args.obs_shape = env.observation_space.shape
    args.action_dim = np.prod(env.action_space.shape)
    

    # env.seed(seed)


    # import configs
    task = args.task.split('-')[0]
    import_path = f"static_fns.{task.lower()}"
    static_fns = importlib.import_module(import_path).StaticFns
    config_path = f"configs.{task.lower()}"
    config = importlib.import_module(config_path).default_config

    # create policy model
    actor_backbone = MLP(input_dim=np.prod(args.obs_shape), hidden_dims=[256, 256])
    critic1_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    critic2_backbone = MLP(input_dim=np.prod(args.obs_shape) + args.action_dim, hidden_dims=[256, 256])
    dist = DiagGaussian(
        latent_dim=getattr(actor_backbone, "output_dim"),
        output_dim=args.action_dim,
        unbounded=True,
        conditioned_sigma=True
    )

    actor = ActorProb(actor_backbone, dist, util.device)
    critic1 = Critic(critic1_backbone, util.device)
    critic2 = Critic(critic2_backbone, util.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    if args.auto_alpha:
        # target_entropy = args.target_entropy if args.target _entropy \
        #     else -np.prod(env.action_space.shape)
        target_entropy = -np.prod(env.action_space.shape)
        args.target_entropy = target_entropy

        log_alpha = torch.zeros(1, requires_grad=True, device=util.device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=args.alpha_lr)
        args.alpha = (target_entropy, log_alpha, alpha_optim)

    # create policy
    sac_policy = SACPolicy(
        actor,
        critic1,
        critic2,
        actor_optim,
        critic1_optim,
        critic2_optim,
        action_space=env.action_space,
        dist=dist,
        tau=args.tau,
        gamma=args.gamma,
        alpha=args.alpha,
        device=util.device
    )

    if "vae" in args.classifier_model_name:
        classifier = VAE(
            # hidden_dims= args.vae_hidden_dims,
            device=util.device
        ).to(util.device)
        classifier_dict = classifier.load_model(args.classifier_model_name)
        print("vae laoded")
    elif "realnvp" in args.classifier_model_name:
        classifier = RealNVP(
        device=util.device
        ).to(util.device)
        classifier_dict = classifier.load_model(args.classifier_model_name)
    elif "kde" in args.classifier_model_name:
        classifier = PercentileThresholdKDE(
        devid=args.devid
        )
        classifier_dict = classifier.load_model(args.classifier_model_name)
    elif "neuralODE" in args.classifier_model_name:
        print("Loading Neural ODE based classifier... for task:", args.task)
        # Use the new NeuralODEOOD.load_model interface
        device = f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu"

        # Load model using NeuralODEOOD wrapper
        classifier_dict = NeuralODEOOD.load_model(
            save_path=args.classifier_model_name.replace('_model.pt', ''),
            target_dim=args.target_dim,
            hidden_dims=(512, 512),
            activation="silu",
            time_dependent=True,
            solver="dopri5",
            t0=0.0,
            t1=1.0,
            rtol=1e-5,
            atol=1e-5,
            device=device
        )
        # classifier_dict now contains: {'model': ood_model, 'threshold': ..., 'mean': ..., 'std': ...}
        # Rename 'threshold' to 'thr' for compatibility with transition_model
        classifier_dict['thr'] = classifier_dict['threshold']
    elif "diffusion" in args.classifier_model_name:
        print("Loading Diffusion based classifier... for task:", args.task)
        # Load model using build_model_from_ckpt from monte_carlo_sampling_unconditional
        device = f"cuda:{args.devid}" if torch.cuda.is_available() else "cpu"
        ckpt_path = args.classifier_model_name
        sched_dir = f"/public/gormpo/models/{args.task.lower().split('_')[0].split('-')[0]}/diffusion/scheduler/scheduler_config.json"

        # Build model
        model, cfg = build_model_from_ckpt(ckpt_path, device)

        # Get target dimension
        ckpt = torch.load(ckpt_path, map_location=device)
        target_dim = ckpt.get("target_dim")

        # Load scheduler
        try:
            scheduler = DDIMScheduler.from_pretrained(sched_dir)
        except Exception:
            try:
                scheduler = DDPMScheduler.from_pretrained(sched_dir)
            except Exception as e:
                print(f"Warning: Could not load scheduler: {e}")
                scheduler = DDIMScheduler(
                    num_train_timesteps=1000,
                    beta_schedule="linear",
                    prediction_type="epsilon",
                )

        # Wrap in our interface
        diffusion_wrapper = DiffusionDensityWrapper(model, scheduler, target_dim, device)

        # Load threshold from metrics if available
        thr_path = f"diffusion/monte_carlo_results/{args.task.lower().split('_')[0].split('-')[0]}_unconditional_ddpm/elbo_metrics.json"
        if os.path.exists(thr_path):
            with open(thr_path, 'r') as f:
                metrics = json.load(f)
            thr = metrics.get("percentile_1.0_logp", 0.0)
        else:
            print(f"Warning: Threshold file not found at {thr_path}, using default threshold 0.0")
            thr = 0.0

        classifier_dict = {'model': diffusion_wrapper, 'thr': thr}
    # create dynamics model
    dynamics_model = TransitionModel(obs_space=env.observation_space,
                                     action_space=env.action_space,
                                     static_fns=static_fns,
                                     lr=args.dynamics_lr,
                                     classifier = classifier_dict,
                                     type = args.penalty_type,
                                     reward_penalty_coef = args.reward_penalty_coef,
                                     **config["transition_params"]
                                     )    
  

    offline_buffer = ReplayBuffer(
    buffer_size=len(dataset["observations"]),
    obs_shape=args.obs_shape,
    obs_dtype=np.float32,
    action_dim=args.action_dim,
    action_dtype=np.float32
    )

    offline_buffer.load_dataset(dataset)    

    model_buffer = ReplayBuffer(
        buffer_size=args.rollout_batch_size * args.rollout_length * args.model_retain_epochs,
        obs_shape=args.obs_shape,
        obs_dtype=np.float32,
        action_dim=args.action_dim,
        action_dtype=np.float32
    )
    
    # create MOPO algo
    algo = MOPO(
        sac_policy,
        dynamics_model,
        offline_buffer=offline_buffer,
        model_buffer=model_buffer,
        reward_penalty_coef=args.reward_penalty_coef,
        rollout_length=args.rollout_length, 
        # rollout_batch_size=args.rollout_batch_size,
        batch_size=args.batch_size,
        real_ratio=args.real_ratio,
        logger=logger,
        **config["mopo_params"]
    )
    #load world model   
    print(len([args.task, args.data_path]))
    print(args.data_path)
    dynamics_model.load_model([args.task, args.data_path]) 

   
    # create trainer
    trainer = Trainer(
        algo,
        eval_env=env,
        epoch=args.epoch,
        step_per_epoch=args.step_per_epoch,
        rollout_freq=args.rollout_freq,
        logger=logger,
        log_freq=args.log_freq,
        run_id = run.id if run!=None else 0,
        env_name = args.task,
        eval_episodes=args.eval_episodes,
        terminal_counter= args.terminal_counter if args.task == "Abiomed-v0" else None,
        
    )

    # pretrain dynamics model on the whole dataset
    # trainer.train_dynamics()
    # 

    # policy_state_dict = torch.load(os.path.join(util.logger_model.log_path, f"policy_{args.task}.pth"))
    # sac_policy.load_state_dict(policy_state_dict)
    # begin train
    trainer.train_policy()

   
    return sac_policy, trainer