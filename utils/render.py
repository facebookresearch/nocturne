"""Render functions to create a video of a traffic scene."""

import os
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np
import pandas as pd
from box import Box
import subprocess
import pickle
from tempfile import mkdtemp
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
import wandb
from nocturne import Action
from nocturne.envs.base_env import BaseEnv


def discretize_action(env_config: Box, action: Action) -> Tuple[Action, int]:
    """Discretize actions."""
    acceleration_actions = np.linspace(
        start=env_config.accel_lower_bound,
        stop=env_config.accel_upper_bound,
        num=env_config.accel_discretization,
    )
    acceleration_idx = np.abs(action.acceleration - acceleration_actions).argmin()
    action.acceleration = acceleration_actions[acceleration_idx]

    steering_actions = np.linspace(
        start=env_config.steering_lower_bound,
        stop=env_config.steering_upper_bound,
        num=env_config.steering_discretization,
    )
    steering_idx = np.abs(action.steering - steering_actions).argmin()
    action.steering = steering_actions[steering_idx]

    action_idx = acceleration_idx * env_config.steering_discretization + steering_idx

    return action, action_idx

def make_video(
    env_config: Box,
    exp_config: Box,
    video_config: Box,
    model: Union[str, OnPolicyAlgorithm],
    n_steps: Optional[int],
    *,
    deterministic: bool = True,
    max_steps: int = 80,
    snap_interval: int = 4,
    frames_per_second: int = 4,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Make a video of a traffic scene. 

    NOTE: Xvfb has a memory leak, which causes the memory usage to continually increase with each video rendered. 
    Our workaround is to run this function in a separate process, and kill the process after the video is rendered.
    More info about the memory leak in Xvfb: http://blog.jeffterrace.com/2012/07/xvfb-memory-leak-workaround.html

    Args:
        env_config (Box): RL environment configuration.
        exp_config (Box): Algo configuration.
        video_config (Box): Rendering configuration.
        model (Union[str, OnPolicyAlgorithm]): Policy to use.
        n_steps (Optional[int]): The global step number. Defaults to None.
        deterministic (bool, optional): If true, set policy to determistic mode. Defaults to True.
        max_steps (int, optional): Episode length. Defaults to 80.
        snap_interval (int, optional): Take snapshot every n steps. Defaults to 4.
        frames_per_second (int, optional): Speed with which to replay video. Defaults to 4.
    """
    if n_steps is not None:
        formatted_global_step = '{:,}'.format(n_steps)
        NUM_VIDEOS = min(env_config.num_files, exp_config.ma_callback.record_n_scenes)
    else:
        formatted_global_step = None
        NUM_VIDEOS = 1

    # If non-deterministic, ensure that the environment is not seeded
    if not deterministic:
        env_config.seed = None

    # Make env
    env_config.sample_file_method = "no_replacement"

    # Write env config and model to pickle
    try:
        temp_dir = Path(mkdtemp())
        with open(temp_dir / "env_config.pkl" , "wb") as env_config_file:
            pickle.dump(obj=env_config, file=env_config_file)
        with open(temp_dir / "video_config.pkl" , "wb") as video_config_file:
            pickle.dump(obj=video_config, file=video_config_file)
        if model not in ("expert", "expert_discrete"):
            with open(temp_dir / "policy_model.pkl", "wb") as model_file:
                pickle.dump(obj=model.policy, file=model_file)

        env = BaseEnv(env_config)  # TODO: Write inputs to temp storage and pass path to render_video.py

        # Record video for specified number of scenes
        for scene_idx in range(NUM_VIDEOS):

            _ = env.reset()

            if model in ("expert", "expert_discrete"):
                wandb_log_keys = [
                    f"actions/agent_{{}}/expert_action_idx",
                    f"actions/agent_{{}}/expert_acceleration",
                    f"actions/agent_{{}}/expert_steering",
                ]
            else:
                wandb_log_keys = [
                f"actions/agent_{{}}/action_idx_{n_steps}",
                f"actions/agent_{{}}/acceleration_{n_steps}",
                f"actions/agent_{{}}/steering_{n_steps}",
            ]

            for agent in env.controlled_vehicles:
                for wandb_log_key in wandb_log_keys:
                    wandb.define_metric(
                        wandb_log_key.format(agent.id), step_metric="timestep"
                    )

            # Render the frames
            subprocess.call(
                [
                    "python",
                    "utils/render_in_subprocess.py", #TODO @Daphne: Change this so that hardcoded path is removed
                    temp_dir,
                    model if isinstance(model, str) else "custom",
                    str(scene_idx),
                    str(max_steps),
                    str(snap_interval),
                    "--deterministic" if deterministic else "--no-deterministic",
                    "--headless-machine" if exp_config.where_am_i == "headless_machine" else "--no-headless-machine",
                ]
            )

            # Load frames from temp pickles
            with open(temp_dir / "frames.pkl", "rb") as movie_frames_file:
                movie_frames = pickle.load(movie_frames_file)

            # Log video to wandb
            video_key = f"Policy | Scene #{scene_idx}" if n_steps is not None else model
            wandb.log(
                {
                    "step": n_steps,
                    video_key: wandb.Video(movie_frames, fps=frames_per_second, caption=f'Global step: {formatted_global_step}'),
                },
            )
        
        env.close()
        env_config.sample_file_method = "random"

    finally:
        # Remove files in temp_dir
        for file in temp_dir.glob("*"):
            if file.is_file():
                os.remove(file)

        # Remove temp_dir
        os.rmdir(temp_dir)
