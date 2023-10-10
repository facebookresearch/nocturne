"""Cast a multi-agent env as vec env. NOTE: only reset if all agents are done and mask invalid samples."""
import logging
from datetime import datetime

import torch
from stable_baselines3 import PPO
import wandb

from utils.config import load_config
from utils.string_utils import datetime_to_str

# Multi-agent as vectorized environment
from utils.sb3_vec_env_SA_w_dead_agents import MultiAgentAsVecEnv

# Custom callback
from utils.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports masking
from algorithms.custom_ppo import CustomPPO

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":

    # Load environment and experiment configurations
    env_config = load_config("env_config")

    # Ensure we're only controlling one agent
    env_config.max_num_vehicles = 1
    exp_config = load_config("exp_config")

    logging.info(f"number of agents == envs == {env_config.max_num_vehicles}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_id = None
    if exp_config.track_wandb:
        # Set up run
        run_id = datetime_to_str(dt=datetime.now())
        run_id = f"{run_id}_s{exp_config.seed}"
        run = wandb.init(
            project=exp_config.project,
            name=run_id,
            config={**exp_config, **env_config},
            id=run_id,
            **exp_config.wandb,
        )

    # Make environment
    env = MultiAgentAsVecEnv(config=env_config, num_envs=env_config.max_num_vehicles)
    
    # Set device
    exp_config.ppo.device = device

    # Initialize custom callback  
    custom_callback = CustomMultiAgentCallback(
        max_agents=env_config.max_num_vehicles,
        env_config=env_config,
        exp_config=exp_config,
    )

    # # Use default PPO class (only works with a single agent)
    # model = PPO(
    #     policy=exp_config.ppo.policy,
    #     env=env,
    #     seed=exp_config.seed,
    #     tensorboard_log=f"runs/{run_id}" if run_id is not None else None,
    #     verbose=1,
    # )

    # Use custom PPO class
    model = CustomPPO(
        n_steps=3500, # Compensate for the decrease in number of samples per rollout
        policy=exp_config.ppo.policy,
        env=env,
        seed=exp_config.seed, # Seed for the pseudo random generators
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