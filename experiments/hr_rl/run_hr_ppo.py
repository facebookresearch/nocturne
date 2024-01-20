"""Train HR-PPO agent."""
import logging
from contextlib import nullcontext
from datetime import datetime
from typing import Callable

import numpy as np
import torch
from stable_baselines3.common.policies import ActorCriticPolicy

import wandb

# Permutation equivariant network
from networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from utils.random_utils import init_seed
from utils.render import make_video

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.reg_ppo import RegularizedPPO
from utils.string_utils import datetime_to_str

logging.basicConfig(level=logging.INFO)


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func


def train(env_config, exp_config, video_config, model_config):  # pylint: disable=redefined-outer-name
    """Train RL agent using PPO."""
    # Ensure reproducability
    init_seed(env_config, exp_config, exp_config.seed)

    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config,
        num_envs=env_config.max_num_vehicles,
    )

    # Set up run
    datetime_ = datetime_to_str(dt=datetime.now())
    run_id = f"{datetime_}" if exp_config.track_wandb else None

    # Add scene to config
    exp_config.scene = env.filename

    with wandb.init(
        project=exp_config.project,
        name=run_id,
        group=exp_config.group,
        config={**exp_config, **env_config},
        id=run_id,
        **exp_config.wandb,
    ) if exp_config.track_wandb else nullcontext() as run:
        # Set device
        exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")
        logging.info(f"Learning in {len(env.env.files)} scene(s): {env.env.files} | using {exp_config.ppo.device}")
        logging.info(f"--- obs_space: {env.observation_space.shape[0]} ---")
        logging.info(f"Action_space\n: {env.env.idx_to_actions}")

        if exp_config.reg_weight > 0.0:
            logging.info(f"Regularization weight: {exp_config.reg_weight} with policy: {exp_config.human_policy_path}")

        # Initialize custom callback
        custom_callback = CustomMultiAgentCallback(
            env_config=env_config,
            exp_config=exp_config,
            video_config=video_config,
            wandb_run=run if run_id is not None else None,
        )

        # Make scene init video to check expert actions
        if exp_config.track_wandb:
            for model in exp_config.wandb_init_videos:
                make_video(
                    env_config=env_config,
                    exp_config=exp_config,
                    video_config=video_config,
                    filenames=[env.filename],
                    model=model,
                    n_steps=None,
                )

        human_policy = None
        # Load human reference policy if regularization is used
        if exp_config.reg_weight > 0.0:
            saved_variables = torch.load(exp_config.human_policy_path, map_location=exp_config.ppo.device)
            human_policy = ActorCriticPolicy(**saved_variables["data"])
            human_policy.load_state_dict(saved_variables["state_dict"])
            human_policy.to(exp_config.ppo.device)

        # Set up PPO
        model = RegularizedPPO(
            learning_rate=linear_schedule(float(exp_config.ppo.learning_rate)),
            reg_policy=human_policy,
            reg_weight=exp_config.reg_weight,  # Regularization weight; lambda
            env=env,
            n_steps=exp_config.ppo.n_steps,
            policy=LateFusionPolicy,
            ent_coef=exp_config.ppo.ent_coef,
            vf_coef=exp_config.ppo.vf_coef,
            seed=exp_config.seed,  # Seed for the pseudo random generators
            verbose=exp_config.verbose,
            tensorboard_log=f"runs/{run_id}" if run_id is not None else None,
            device=exp_config.ppo.device,
            env_config=env_config,
            mlp_class=LateFusionNet,
            mlp_config=model_config,
        )

        # Log number of trainable parameters
        policy_params = filter(lambda p: p.requires_grad, model.policy.parameters())
        params = sum(np.prod(p.size()) for p in policy_params)
        exp_config.n_policy_params = params
        logging.info(f"Policy | trainable params: {params:,} \n")

        # Learn
        model.learn(
            **exp_config.learn,
            callback=custom_callback,
        )


if __name__ == "__main__":
    # Load environment and experiment configurations
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    video_config = load_config("video_config")

    # Train
    train(
        env_config=env_config,
        exp_config=exp_config,
        video_config=video_config,
        model_config=None,
    )
