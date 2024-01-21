import glob
import logging
import os
import pickle
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString
from tqdm import tqdm

from nocturne.envs.base_env import BaseEnv
from utils.config import load_config


def step_through_scene(env, mode, filename=None, num_steps=90):
    # Reset env
    if filename is None:
        try:
            obs_dict = env.reset()
        except ValueError:
            return np.zeros((1, num_steps, 2))  # Return empty array if no agents
    else:
        try:
            obs_dict = env.reset(filename)
        except ValueError:
            return np.zeros((1, num_steps, 2))  # Return empty array if no agents

    num_agents = len(env.controlled_vehicles)
    # Storage
    agent_positions = np.full(fill_value=np.nan, shape=(num_agents, num_steps, 2))
    agent_speed = np.full(fill_value=np.nan, shape=(num_agents, num_steps))
    goal_achieved, veh_edge_collision, veh_veh_collision = (
        np.zeros(num_agents),
        np.zeros(num_agents),
        np.zeros(num_agents),
    )

    # Make sure the agent ids are in the same order
    agent_ids = np.sort([veh.id for veh in env.controlled_vehicles])
    agent_id_to_idx_dict = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
    last_info_dicts = {agent_id: {} for agent_id in agent_ids}
    dead_agent_ids = []

    # Set control mode
    if mode == "expert":
        for obj in env.controlled_vehicles:
            obj.expert_control = True
    if mode == "policy":
        for obj in env.controlled_vehicles:
            obj.expert_control = False

    # Step through scene
    for timestep in range(num_steps):
        # Get actions
        if mode == "expert":
            for veh_obj in env.controlled_vehicles:
                if veh_obj.id not in dead_agent_ids:
                    veh_idx = agent_id_to_idx_dict[veh_obj.id]
                    agent_positions[veh_idx, timestep] = np.array([veh_obj.position.x, veh_obj.position.y])
                    agent_speed[veh_idx, timestep] = veh_obj.speed

        action_dict = {}

        # Step env
        obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

        # Update dead agents based on most recent done_dict
        for agent_id, is_done in done_dict.items():
            if is_done and agent_id not in dead_agent_ids:
                if agent_id != "__all__":
                    dead_agent_ids.append(agent_id)

                    # Store agents' last info dict
                    last_info_dicts[agent_id] = info_dict[agent_id].copy()

        if done_dict["__all__"]:
            for agent_id in agent_ids:
                agent_idx = agent_id_to_idx_dict[agent_id]
                goal_achieved[agent_idx] += last_info_dicts[agent_id]["goal_achieved"]
                veh_edge_collision[agent_idx] += last_info_dicts[agent_id]["veh_edge_collision"]
                veh_veh_collision[agent_idx] += last_info_dicts[agent_id]["veh_veh_collision"]
            break

    return agent_positions


def plot_lines(line1, line2, title):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_title(title)
    ax.plot(*line1.xy)
    ax.plot(*line2.xy)
    fig.show()


def create_intersecting_path_dict(env, traffic_scenes, save_as="int_paths"):
    scene_intersecting_paths_dict = {}

    for traffic_scene in tqdm(traffic_scenes):
        expert_trajectories = step_through_scene(env, filename=traffic_scene, mode="expert")

        num_vehicles = expert_trajectories.shape[0]
        iterable = list(range(num_vehicles))
        n = 2

        # Get all possible combinations
        veh_combinations = list(combinations(iterable, n))
        num_intersecting_paths = 0

        for veh_i, veh_j in veh_combinations:
            path_veh_i = expert_trajectories[veh_i, :, :]
            path_veh_j = expert_trajectories[veh_j, :, :]

            # Filter out nans
            nonnan_ids = np.logical_not(
                np.logical_or(
                    np.isnan(path_veh_i),
                    np.isnan(path_veh_j),
                )
            )
            new_dim = int(len(path_veh_i[nonnan_ids]) // 2)

            if nonnan_ids.sum() > 2:
                # Convert to line objects
                line1 = LineString(path_veh_i[nonnan_ids].reshape(new_dim, 2))
                line2 = LineString(path_veh_j[nonnan_ids].reshape(new_dim, 2))

            title = "no"
            if line1.intersects(line2):
                num_intersecting_paths += 1
                title = "intersect!"
                # print('lines_intersect!')
                # plot_lines(line1, line2, title=title)

        # Store scene information
        scene_intersecting_paths_dict[traffic_scene] = {}
        scene_intersecting_paths_dict[traffic_scene]["intersecting_paths"] = num_intersecting_paths
        scene_intersecting_paths_dict[traffic_scene]["num_agents"] = num_vehicles

        # print(f'scene: {env.file} has {num_intersecting_paths} intersecting path(s)')

    with open(f"{save_as}.pkl", "wb") as f:
        pickle.dump(scene_intersecting_paths_dict, f)

    return pd.DataFrame(scene_intersecting_paths_dict)


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.DEBUG)

    env_config = load_config("env_config")
    exp_config = load_config("exp_config")

    MAX_FILES = 2

    # Train
    file_paths = glob.glob(f"{env_config.data_path}" + "/tfrecord*")
    eval_files = sorted([os.path.basename(file) for file in file_paths])[:MAX_FILES]

    env = BaseEnv(env_config)

    agent_positions, agent_speed, goal_achieved, veh_edge_collision, veh_veh_collision = _step_through_scene(
        env, eval_files, mode="expert"
    )

    print("DOne")
