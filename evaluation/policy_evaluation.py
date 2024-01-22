"""Evaluate a policy on a set of scenes."""
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nocturne.envs.base_env import BaseEnv
from utils.config import load_config


def evaluate_policy(
    env_config,
    mode,
    controlled_agents,
    data_path,
    select_from_k_scenes=100,
    num_episodes=100,
    scene_path_mapping=None,
    policy=None,
    deterministic=True,
):
    """Evaluate a policy on a set of scenes.

    Args:
    ----
        env_config: Environment configuration.
        mode: Mode of evaluation. Can be one of the following:
            - policy: Evaluate a policy.
            - expert_replay: Replay expert actions.
            - cont_expert_act_replay: Replay continuous expert actions.
            - disc_expert_act_replay: Replay discretized expert actions.
        controlled_agents: Number of agents to control.
        data_path: Path to data.
        select_from_k_scenes: Number of scenes to select from.
        num_episodes: Number of episodes to run; how many times to reset the environment.
        scene_path_mapping (optional): Mapping from scene to dict with the number of intersecting paths of that scene.
        policy (optional): Policy to evaluate.
        deterministic (optional): Whether to use a deterministic policy.

    Raises:
    ------
        ValueError: If scene_path_mapping is provided, if scene is not found in scene_path_mapping.

    Returns:
    -------
        df: performance per scene and vehicle id.
    """
    # Set the number of vehicles to control per scene
    env_config.max_num_vehicles = controlled_agents

    # Set path where to load scenes from
    env_config.data_path = data_path

    # Set which files to use
    env_config.num_files = select_from_k_scenes

    # Make env
    env = BaseEnv(env_config)

    # Storage
    df = pd.DataFrame(
        columns=[
            "scene_id",
            "veh_id",
            "goal_rate",
            "off_road",
            "veh_veh_collision",
        ],
    )

    # Run
    obs_dict = env.reset()
    agent_ids = list(obs_dict.keys())
    veh_id_to_idx = {veh_id: idx for idx, veh_id in enumerate(agent_ids)}
    dead_agent_ids = []
    last_info_dicts = {agent_id: {} for agent_id in agent_ids}
    goal_achieved = np.zeros(len(agent_ids))
    off_road = np.zeros(len(agent_ids))
    veh_veh_coll = np.zeros(len(agent_ids))

    for _ in tqdm(range(num_episodes)):
        logging.debug(f"scene: {env.file} -- veh_id = {agent_ids} --")

        for time_step in range(env_config.episode_length):
            # Get actions
            action_dict = {}

            if mode == "policy" and policy is not None:
                for agent_id in obs_dict:
                    # Get observation
                    obs = torch.from_numpy(obs_dict[agent_id]).unsqueeze(dim=0)

                    # Get action
                    action, _ = policy.predict(obs, deterministic=deterministic)
                    action_dict[agent_id] = int(action)

            elif mode == "policy" and policy is None:
                raise ValueError("Policy is not given. Please provide a policy.")

            if mode == "expert_replay":
                # Use expert actions
                for veh_obj in env.controlled_vehicles:
                    veh_obj.expert_control = True

            if mode == "cont_expert_act_replay":  # Use continuous expert actions
                for veh_obj in env.controlled_vehicles:
                    veh_obj.expert_control = False

                    # Get (continuous) expert action
                    expert_action = env.scenario.expert_action(veh_obj, time_step)

                    action_dict[veh_obj.id] = expert_action

            if mode == "disc_expert_act_replay":  # Use discretized expert actions
                # Get expert actions and discretize
                for veh_obj in env.controlled_vehicles:
                    veh_obj.expert_control = False

                    # Get (continuous) expert action
                    expert_action = env.scenario.expert_action(veh_obj, time_step)

                    # Discretize expert action
                    if expert_action is None:
                        logging.info(f"None at {time_step} for veh {veh_obj.id} in {env.file} \n")

                    elif expert_action is not None:
                        expert_accel, expert_steering, _ = expert_action.numpy()

                        # Map actions to nearest grid indices and joint action
                        acc_grid_idx = np.argmin(np.abs(env.accel_grid - expert_accel))
                        ste_grid_idx = np.argmin(np.abs(env.steering_grid - expert_steering))

                        expert_action_idx = env.actions_to_idx[
                            env.accel_grid[acc_grid_idx],
                            env.steering_grid[ste_grid_idx],
                        ][0]

                        action_dict[veh_obj.id] = expert_action_idx

                        logging.debug(
                            f"true_exp_acc = {expert_action.acceleration:.4f}; "
                            f"true_exp_steer = {expert_action.steering:.4f}"
                        )
                        logging.debug(
                            f"disc_exp_acc = {env.accel_grid[acc_grid_idx]:.4f}; "
                            f"disc_exp_steer = {env.steering_grid[ste_grid_idx]:.4f} \n"
                        )

            # Take a step
            obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

            for agent_id, is_done in done_dict.items():
                if is_done and agent_id not in dead_agent_ids:
                    dead_agent_ids.append(agent_id)
                    # Store agents' last info dict
                    last_info_dicts[agent_id] = info_dict[agent_id].copy()

            if done_dict["__all__"]:
                # Update df
                for agent_id in agent_ids:
                    agend_idx = veh_id_to_idx[agent_id]
                    veh_veh_coll[agend_idx] += last_info_dicts[agent_id]["veh_veh_collision"] * 1
                    off_road[agend_idx] += last_info_dicts[agent_id]["veh_edge_collision"] * 1
                    goal_achieved[agend_idx] += last_info_dicts[agent_id]["goal_achieved"] * 1

                if scene_path_mapping is not None:
                    if env.file in scene_path_mapping:
                        df_scene_i = pd.DataFrame(
                            {
                                "scene_id": env.file,
                                "veh_id": agent_ids,
                                "goal_rate": goal_achieved,
                                "off_road": off_road,
                                "veh_veh_collision": veh_veh_coll,
                                "num_total_vehs": scene_path_mapping[env.file]["num_agents"],
                                "num_controlled_vehs": len(agent_ids),
                                "num_int_paths": scene_path_mapping[env.file]["intersecting_paths"],
                            },
                            index=list(range(len(agent_ids))),
                        )
                    else:
                        raise ValueError(f"Scene {env.file} not found in scene_path_mapping")

                else:
                    df_scene_i = pd.DataFrame(
                        {
                            "scene_id": env.file,
                            "veh_id": agent_ids,
                            "goal_rate": goal_achieved,
                            "off_road": off_road,
                            "veh_veh_collision": veh_veh_coll,
                        },
                        index=list(range(len(agent_ids))),
                    )

                # Append to df
                df = pd.concat([df, df_scene_i], ignore_index=True)

                # Reset
                obs_dict = env.reset()
                agent_ids = list(obs_dict.keys())
                veh_id_to_idx = {veh_id: idx for idx, veh_id in enumerate(agent_ids)}
                dead_agent_ids = []
                last_info_dicts = {agent_id: {} for agent_id in agent_ids}
                goal_achieved = np.zeros(len(agent_ids))
                off_road = np.zeros(len(agent_ids))
                veh_veh_coll = np.zeros(len(agent_ids))

                break  # Proceed to next scene

    return df


if __name__ == "__main__":
    # Global setting
    logger = logging.getLogger()
    logging.basicConfig(format="%(message)s")
    logger.setLevel("INFO")

    env_config = load_config("env_config")

    df_disc_expert_replay = evaluate_policy(
        env_config=env_config,
        data_path=env_config.data_path,
        mode="disc_expert_replay",
        select_from_k_scenes=100,
        num_episodes=100,
        controlled_agents=2,
    )
