"""Cast a multi-agent env as vec env to use SB3's PPO."""
import logging
from datetime import datetime

import torch
import wandb
import numpy as np

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from utils.render import make_video

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.custom_ppo import MultiAgentPPO
from utils.string_utils import datetime_to_str

logging.basicConfig(level=logging.INFO)

def train(env_config, exp_config, video_config):
    """Train RL agent using PPO."""

    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles,
        train_on_single_scene=exp_config.train_on_single_scene,
    )
    # Set up wandb
    RUN_ID = None
    if exp_config.track_wandb:
        # Set up run
        datetime_ = datetime_to_str(dt=datetime.now())
        RUN_ID = f"{exp_config.exp_name}_{datetime_}"
    
        # Add scene to config
        exp_config.scene = env.filename

        run = wandb.init(
            project=exp_config.project,
            name=RUN_ID,
            config={**exp_config, **env_config},
            id=RUN_ID,
            **exp_config.wandb,
        )

    logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")
    logging.info(f"Learning in {env_config.num_files} scene(s): {env.env.files}")
    logging.info(f"--- obs_space: {env.observation_space.shape[0]} ---")
    logging.info(f"Action_space\n: {env.env.idx_to_actions}")

    # Set device
    exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize custom callback
    custom_callback = CustomMultiAgentCallback(
        env_config=env_config,
        exp_config=exp_config,
        video_config=video_config,
        wandb_run=run if RUN_ID is not None else None,
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

    # Set up PPO model
    model = MultiAgentPPO(
        env=env,
        n_steps=exp_config.ppo.n_steps,
        policy=exp_config.ppo.policy,
        ent_coef=exp_config.ppo.ent_coef,
        vf_coef=exp_config.ppo.vf_coef,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        tensorboard_log=f"runs/{RUN_ID}" if RUN_ID is not None else None,
        verbose=exp_config.verbose,
        device=exp_config.ppo.device,
    )

    # Learn
    model.learn(
        **exp_config.learn,
        callback=custom_callback,
    )

    # Finish
    if exp_config.track_wandb:
        run.finish()

if __name__ == "__main__":

    # Load environment and experiment configurations
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    video_config = load_config("video_config")

    # Run
    exp_config.seed = int(np.random.choice([42, 6, 0], size=1))
    train(env_config, exp_config, video_config) 
