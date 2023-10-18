"""Cast a multi-agent env as vec env to use SB3's PPO."""
import logging
from datetime import datetime

import torch

import wandb

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.custom_ppo import MultiAgentPPO
from utils.string import datetime_to_str

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    # Load environment and experiment configurations
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    video_config = load_config("video_config")

    logging.info(f"Loaded Nocturne environment, experiment, and video configurations.")

    # Set the maximum number of agents to control
    # TODO: remove hardcoded var
    env_config.max_num_vehicles = 3

    # Set up wandb
    run_id = None
    if exp_config.track_wandb:
        # Set up run
        run_id = datetime_to_str(dt=datetime.now())
        run_id = f"M={env_config.max_num_vehicles}_S{exp_config.seed}_nsteps={exp_config.ppo.n_steps}"
        run = wandb.init(
            project=exp_config.project,
            name=run_id,
            config={**exp_config, **env_config},
            id=run_id,
            **exp_config.wandb,
        )

    # Make environment
    env = MultiAgentAsVecEnv(config=env_config, num_envs=env_config.max_num_vehicles)

    logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")

    # Set device
    exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize custom callback
    custom_callback = CustomMultiAgentCallback(
        env_config=env_config,
        exp_config=exp_config,
        video_config=video_config,
    )

    # Use custom PPO class
    model = MultiAgentPPO(
        n_steps=exp_config.ppo.n_steps,
        policy=exp_config.ppo.policy,
        env=env,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        tensorboard_log=f"runs/{run_id}" if run_id is not None else None,
        verbose=1,
    )

    # Learn
    model.learn(
        **exp_config.learn,
        callback=custom_callback,
    )

    # Finish
    if exp_config.track_wandb:
        run.finish()
