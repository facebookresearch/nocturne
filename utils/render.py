"""Render functions to create a video of a traffic scene."""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from box import Box
from pyvirtualdisplay import Display
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

    Returns:
        Tuple[np.ndarray, pd.DataFrame]: Movie frames and action dataframe.
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
    env = BaseEnv(env_config)

    # Record video for specified number of scenes
    for scene_idx in range(NUM_VIDEOS):

        next_obs_dict = env.reset()
        next_done_dict = {agent_id: False for agent_id in next_obs_dict}

        frames = []
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

        action_df = pd.DataFrame()
        for timestep in range(max_steps):
            action_dict = {}
            for agent in env.controlled_vehicles:
                if agent.id in next_obs_dict and not next_done_dict[agent.id]:
                    if model in ("expert", "expert_discrete"):
                        agent.expert_control = True
                        action = env.scenario.expert_action(agent, timestep)
                        agent.expert_control = False
                        if action is not None:
                            acceleration, steering = action.acceleration, action.steering
                        if model == "expert_discrete":
                            if action is not None:
                                action, action_idx = discretize_action(env_config=env_config, action=action)
                                acceleration, steering = env.idx_to_actions[action_idx.item()]
                        action_dict[agent.id] = action
                        if model == "expert":
                            agent.expert_control = True
                            action_dict = {}
                    else:
                        obs_tensor = torch.Tensor(next_obs_dict[agent.id]).unsqueeze(dim=0)
                        with torch.no_grad():
                            action_idx, _ = model.predict(obs_tensor, deterministic=deterministic)
                        action_dict[agent.id] = action_idx.item()
                        acceleration, steering = env.idx_to_actions[action_idx.item()]

                    # Log actions per timestep
                    if exp_config.ma_callback.log_agent_actions:
                        if model != "expert":
                            wandb.log({ 
                                wandb_log_keys[0].format(agent.id): action_idx,
                                wandb_log_keys[1].format(agent.id): acceleration, 
                                wandb_log_keys[2].format(agent.id): steering,
                                    "timestep": timestep,
                            })

            next_obs_dict, _, next_done_dict, _ = env.step(action_dict)

            if model in ("expert", "expert_discrete"):
                action_dict = {
                    agent_id: discretize_action(env_config, action)[1] for agent_id, agent in action_dict.items() if action is not None
                }
            action_df = pd.concat(
                (action_df, pd.DataFrame(action_dict, index=[timestep])),
                axis=0,
                ignore_index=True,
            )

            if timestep % snap_interval == 0:
                # If we're on a headless machine: activate display and render
                if exp_config.where_am_i == "headless_machine":
                    with Display() as disp:
                        render_scene = env.scenario.getImage(**video_config.render)
                        frames.append(render_scene.T)
                else:
                    render_scene = env.scenario.getImage(**video_config.render)
                    frames.append(render_scene.T)

            if next_done_dict["__all__"]:
                break

        movie_frames = np.array(frames, dtype=np.uint8)

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

    return movie_frames, action_df
