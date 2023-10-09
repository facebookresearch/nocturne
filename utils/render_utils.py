
"""Utility functions to create a video of a traffic scene."""

import logging
import os
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
from box import Box
from nocturne import Action
from pyvirtualdisplay import Display
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm

import wandb
from nocturne.envs.base_env import BaseEnv
from utils.config import load_config


def discretize_action(env_config: Box, action: Action) -> Tuple[Action, int]:
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

def save_nocturne_video(
    env_config: Box,
    exp_config: Box,
    video_config: Box,
    model: Union[str, OnPolicyAlgorithm],
    n_steps: Optional[int],
    *,
    log_wandb: bool = True,
    deterministic: bool = True,
    max_steps: int = 80,
    frame_rate: int = 4,
    frames_per_second: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame]:

    total_steps = exp_config.learn.total_timesteps
    video_name = (
        model
        if isinstance(model, str)
        else f"Steps {str(n_steps).rjust(len(str(total_steps)), ' ')}"
    )

    # If non-deterministic, ensure that the environment is not seeded
    if not deterministic:
        env_config.seed = None

    # Make env
    env = BaseEnv(env_config)
    next_obs_dict = env.reset()
    next_done_dict = {agent_id: False for agent_id in next_obs_dict}

    frames = []
    # if log_wandb:
    #     wandb_log_keys = [
    #         f"agent_{{}}/action_idx/{video_name}",
    #         f"agent_{{}}/acceleration/{video_name}",
    #         f"agent_{{}}/steering/{video_name}",
    #     ]
    #     for agent in env.controlled_vehicles:
    #         for wandb_log_key in wandb_log_keys:
    #             wandb.define_metric(
    #                 wandb_log_key.format(agent.id), step_metric="timestep"
    #             )

    action_df = pd.DataFrame()
    for timestep in range(max_steps):
        action_dict = {}
        for agent in env.controlled_vehicles:
            if agent.id in next_obs_dict and not next_done_dict[agent.id]:
                if model in ("expert", "expert_discrete"):
                    agent.expert_control = True
                    action = env.scenario.expert_action(agent, timestep)
                    agent.expert_control = False
                    if model == "expert_discrete":
                        action, action_idx = discretize_action(
                            env_config=env_config, action=action
                        )
                    action_dict[agent.id] = action
                    acceleration, steering = action.acceleration, action.steering
                else:
                    obs_tensor = torch.Tensor(next_obs_dict[agent.id]).unsqueeze(dim=0)
                    with torch.no_grad():
                        if isinstance(model, BehavioralCloningAgentJoint):
                            action_idx, _ = model(obs_tensor)
                        else:
                            action_idx, _ = model.predict(
                                obs_tensor, deterministic=deterministic
                            )
                    action_dict[agent.id] = action_idx.item()
                    acceleration, steering = env.idx_to_actions[action_idx.item()]
                # if log_wandb:
                #     wandb.log(
                #         {
                #             wandb_log_keys[0].format(agent.id): None
                #             if model == "expert"
                #             else action_idx,
                #             wandb_log_keys[1].format(agent.id): acceleration,
                #             wandb_log_keys[2].format(agent.id): steering,
                #             "timestep": timestep,
                #         },
                #     )
        next_obs_dict, _, next_done_dict, _ = env.step(action_dict)
        if model in ("expert", "expert_discrete"):
            action_dict = {
                agent_id: discretize_action(env_config, action)[1]
                for agent_id, agent in action_dict.items()
            }
        action_df = pd.concat(
            (action_df, pd.DataFrame(action_dict, index=[timestep])),
            axis=0,
            ignore_index=True,
        )

        if timestep % frame_rate == 0:
            # Activate display and render
            with Display() as disp:
                render_scene = env.scenario.getImage(**video_config)
                frames.append(render_scene.T)

        if next_done_dict["__all__"]:
            break

    movie_frames = np.array(frames, dtype=np.uint8)

    if log_wandb:
        video_key = video_name if n_steps is None else "video"

        wandb.log(
            {
                "step": n_steps,
                video_key: wandb.Video(
                    movie_frames, fps=frames_per_second, caption=video_name
                ),
            },
        )
    env.close()

    return movie_frames, action_df