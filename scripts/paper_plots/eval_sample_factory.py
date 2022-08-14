# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run a policy over the entire train set.

TODO(ev) refactor, this is wildly similar to visualize_sample_factory
"""

from copy import deepcopy
from collections import deque, defaultdict
import itertools
from itertools import repeat
import json
import multiprocessing as mp
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution, \
     CategoricalActionDistribution
from sample_factory.algorithms.utils.arguments import load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from examples.sample_factory_files.run_sample_factory import register_custom_components

from cfgs.config import PROCESSED_VALID_NO_TL, PROCESSED_TRAIN_NO_TL, \
    ERR_VAL, set_display_window

CB_color_cycle = [
    '#377eb8', '#ff7f00', '#4daf4a', '#f781bf', '#a65628', '#984ea3',
    '#999999', '#e41a1c', '#dede00'
]


class Bunch(object):
    """Converts a dict into an object with the keys as attributes."""

    def __init__(self, adict):
        self.__dict__.update(adict)


def ccw(A, B, C):
    """Blah."""
    return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])


def intersect(A, B, C, D):
    """Check if two line segments AB and CD intersect."""
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)


def poly_intersection(poly1, poly2):
    """Compute if two polylines intersect."""
    for i, p1_first_point in enumerate(poly1[:-1]):
        p1_second_point = poly1[i + 1]

        for j, p2_first_point in enumerate(poly2[:-1]):
            p2_second_point = poly2[j + 1]

            if intersect(p1_first_point, p1_second_point, p2_first_point,
                         p2_second_point):
                return True

    return False


def run_rollouts(env,
                 cfg,
                 device,
                 expert_trajectory_dict,
                 distance_bins,
                 intersection_bins,
                 veh_intersection_dict,
                 actor_1,
                 actor_2=None):
    """Run a single rollout.

    Args:
        env (_type_): Env we are running.
        cfg (dict): dictionary configuring the environment.
        device (str): device you want to run the model on
        expert_trajectory_dict (dict[str]: np.array): expert trajectories
            keyed by ID
        distance_bins (np.array): bins used to compute the goal
            rate as a function of the starting distance from goal
        intersection_bins (np.array): bins used to compute the
            goal rate as a function of the number of intersections
            between paths in the expert trajectories
        veh_intersection_dict (dict[str]: np.array): dict mapping
            a vehicle ID to the number of intersections it
            experienced
        actor_1: SampleFactory agent
        actor_2: SampleFactory agent. Will be none unless we're testing for
                ZSC

    Returns
    -------
        avg_goal: average goal rate of agents
        avg_collisions: average collision rate of agents
        avg_veh_edge_collisions: average veh-edge collision rate
        avg_veh_veh_collisions: average veh-veh collision rate
        success_rate_by_distance: np.array(number of distance bins, 4)
            where the row indexes how far the vehicle was from goal
            at initialization and where the column index is
            [goal rate, collision rate, veh-veh collision rate, counter of
                            number of vehicles in this bin]
        success_rate_by_num_agents: np.array(maximum number of vehicles, 4)
            where the row index is how many vehicles were in this episode
            where the column index is [goal rate, collision rate,
                            veh-veh collision rate, counter of
                            number of vehicles in this bin]
        success_rate_by_intersections: np.array(number of intersections, 4)
            where the row index is how many intersections that vehicle
            had and where the column index is [goal rate, collision rate,
                            veh-veh collision rate, counter of
                            number of vehicles in this bin]
        np.mean(ades): mean average displacement error of all vehicles in the
                       episode
        np.mean(fdes): mean final displacement error of all vehicles in the
                       episode
        veh_counter(int): how many vehicles were in that episode
    """
    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    obs = env.reset()
    rollout_traj_dict = defaultdict(lambda: np.zeros((80, 2)))
    # some key information for tracking statistics
    goal_dist = env.goal_dist_normalizers
    valid_indices = env.valid_indices
    agent_id_to_env_id_map = env.agent_id_to_env_id_map
    env_id_to_agent_id_map = env.env_id_to_agent_id_map

    success_rate_by_num_agents = np.zeros((cfg.max_num_vehicles, 4))
    success_rate_by_distance = np.zeros((distance_bins.shape[-1], 4))
    success_rate_by_intersections = np.zeros((intersection_bins.shape[-1], 4))
    if actor_2 is not None:
        # pick which valid indices go to which policy
        val = np.random.uniform()
        if val < 0.5:
            num_choice = int(np.floor(len(valid_indices) / 2.0))
        else:
            num_choice = int(np.ceil(len(valid_indices) / 2.0))
        indices_1 = list(
            np.random.choice(valid_indices, num_choice, replace=False))
        indices_2 = [val for val in valid_indices if val not in indices_1]
        rnn_states = torch.zeros(
            [env.num_agents, get_hidden_size(cfg)],
            dtype=torch.float32,
            device=device)
        rnn_states_2 = torch.zeros(
            [env.num_agents, get_hidden_size(cfg)],
            dtype=torch.float32,
            device=device)
    else:
        rnn_states = torch.zeros(
            [env.num_agents, get_hidden_size(cfg)],
            dtype=torch.float32,
            device=device)
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents
    goal_achieved = [False] * len(valid_indices)
    collision_observed = [False] * len(valid_indices)
    veh_veh_collision_observed = [False] * len(valid_indices)
    veh_counter = 0

    while not all(finished_episode):
        with torch.no_grad():
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            # we have to make a copy before doing the pass
            # because (for some reason), sample factory is making
            # some changes to the obs in the forwards pass
            # TBD what it is
            if actor_2 is not None:
                obs_torch_2 = deepcopy(obs_torch)
                policy_outputs_2 = actor_2(obs_torch_2,
                                           rnn_states_2,
                                           with_action_distribution=True)

            policy_outputs = actor_1(obs_torch,
                                     rnn_states,
                                     with_action_distribution=True)

            # sample actions from the distribution by default
            # also update the indices that should be drawn from the second policy
            # with its outputs
            actions = policy_outputs.actions
            if actor_2 is not None:
                actions[indices_2] = policy_outputs_2.actions[indices_2]

            action_distribution = policy_outputs.action_distribution
            if isinstance(action_distribution, ContinuousActionDistribution):
                if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                    actions = action_distribution.means
                    if actor_2 is not None:
                        actions[
                            indices_2] = policy_outputs_2.action_distribution.means[
                                indices_2]
            if isinstance(action_distribution, CategoricalActionDistribution):
                if not cfg.discrete_actions_sample:
                    actions = policy_outputs['action_logits'].argmax(axis=1)
                    if actor_2 is not None:
                        actions[indices_2] = policy_outputs_2[
                            'action_logits'].argmax(axis=1)[indices_2]

            actions = actions.cpu().numpy()

            for veh in env.unwrapped.get_objects_that_moved():
                # only check vehicles we are actually controlling
                if veh.expert_control is False:
                    rollout_traj_dict[veh.id][
                        env.step_num] = veh.position.numpy()
                if int(veh.collision_type) == 1:
                    if veh.getID() in env_id_to_agent_id_map.keys():
                        agent_id = env_id_to_agent_id_map[veh.getID()]
                        idx = valid_indices.index(agent_id)
                        veh_veh_collision_observed[idx] = 1

            rnn_states = policy_outputs.rnn_states
            if actor_2 is not None:
                rnn_states_2 = policy_outputs_2.rnn_states

            obs, rew, done, infos = env.step(actions)
            episode_reward += rew

            for i, index in enumerate(valid_indices):
                goal_achieved[
                    i] = infos[index]['goal_achieved'] or goal_achieved[i]
                collision_observed[
                    i] = infos[index]['collided'] or collision_observed[i]

            for agent_i, done_flag in enumerate(done):
                if done_flag:
                    finished_episode[agent_i] = True
                    episode_rewards[agent_i].append(episode_reward[agent_i])
                    true_rewards[agent_i].append(infos[agent_i].get(
                        'true_reward', episode_reward[agent_i]))
                    log.info(
                        'Episode finished for agent %d. Reward: %.3f, true_reward: %.3f',
                        agent_i, episode_reward[agent_i],
                        true_rewards[agent_i][-1])
                    rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)],
                                                      dtype=torch.float32,
                                                      device=device)
                    episode_reward[agent_i] = 0

            if all(finished_episode):
                avg_episode_rewards_str, avg_true_reward_str = '', ''
                for agent_i in range(env.num_agents):
                    avg_rew = np.mean(episode_rewards[agent_i])
                    avg_true_rew = np.mean(true_rewards[agent_i])
                    if not np.isnan(avg_rew):
                        if avg_episode_rewards_str:
                            avg_episode_rewards_str += ', '
                        avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                    if not np.isnan(avg_true_rew):
                        if avg_true_reward_str:
                            avg_true_reward_str += ', '
                        avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'
                avg_goal = infos[0]['episode_extra_stats']['goal_achieved']
                avg_collisions = infos[0]['episode_extra_stats']['collided']
                avg_veh_edge_collisions = infos[0]['episode_extra_stats'][
                    'veh_edge_collision']
                avg_veh_veh_collisions = infos[0]['episode_extra_stats'][
                    'veh_veh_collision']
                success_rate_by_num_agents[len(valid_indices) - 1,
                                           0] += avg_goal
                success_rate_by_num_agents[len(valid_indices) - 1,
                                           1] += avg_collisions
                success_rate_by_num_agents[len(valid_indices) - 1,
                                           2] += np.mean(
                                               veh_veh_collision_observed)
                success_rate_by_num_agents[len(valid_indices) - 1, 3] += 1
                # track how well we do as a function of distance
                for i, index in enumerate(valid_indices):
                    env_id = agent_id_to_env_id_map[index]
                    bin = np.searchsorted(distance_bins, goal_dist[env_id])
                    success_rate_by_distance[bin - 1, :] += [
                        goal_achieved[i], collision_observed[i],
                        veh_veh_collision_observed[i], 1
                    ]
                # track how well we do as number of intersections
                for i, index in enumerate(valid_indices):
                    env_id = agent_id_to_env_id_map[index]
                    bin = min(veh_intersection_dict[env_id],
                              distance_bins.shape[-1] - 1)
                    success_rate_by_intersections[bin, :] += [
                        goal_achieved[i], collision_observed[i],
                        veh_veh_collision_observed[i], 1
                    ]
                # compute ADE and FDE
                ades = []
                fdes = []
                for agent_id, traj in rollout_traj_dict.items():
                    masking_arr = traj.sum(axis=1)
                    mask = (masking_arr != 0.0) * (masking_arr !=
                                                   traj.shape[1] * ERR_VAL)
                    expert_mask_arr = expert_trajectory_dict[agent_id].sum(
                        axis=1)
                    expert_mask = (expert_mask_arr != 0.0) * (
                        expert_mask_arr != traj.shape[1] * ERR_VAL)
                    ade = np.linalg.norm(traj -
                                         expert_trajectory_dict[agent_id],
                                         axis=-1)[mask * expert_mask]
                    ades.append(ade.mean())
                    fde = np.linalg.norm(
                        traj - expert_trajectory_dict[agent_id],
                        axis=-1)[np.max(np.argwhere(mask * expert_mask))]
                    fdes.append(fde)
                    veh_counter += 1

                log.info('Avg episode rewards: %s, true rewards: %s',
                         avg_episode_rewards_str, avg_true_reward_str)
                log.info(
                    'Avg episode reward: %.3f, avg true_reward: %.3f',
                    np.mean([
                        np.mean(episode_rewards[i])
                        for i in range(env.num_agents)
                    ]),
                    np.mean([
                        np.mean(true_rewards[i]) for i in range(env.num_agents)
                    ]))

                return (avg_goal, avg_collisions, avg_veh_edge_collisions,
                        avg_veh_veh_collisions, success_rate_by_distance,
                        success_rate_by_num_agents,
                        success_rate_by_intersections, np.mean(ades),
                        np.mean(fdes), veh_counter)


def run_eval(cfgs,
             test_zsc,
             output_path,
             scenario_dir,
             files,
             file_type,
             device='cuda'):
    """Eval a stored agent over all files in validation set.

    Args:
        cfg (dict): configuration file for instantiating the agents and environment.
        test_zsc (bool): if true, we play all agents against all agents
        num_file_loops (int): how many times to loop over the file set

    Returns
    -------
        None: None
    """
    actor_critics = []
    if not isinstance(cfgs, list):
        cfgs = [cfgs]
    for i, cfg in enumerate(cfgs):
        if not isinstance(cfg, Bunch):
            cfg = Bunch(cfg)
        cfg = load_from_checkpoint(cfg)

        render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
        if render_action_repeat is None:
            log.warning('Not using action repeat!')
            render_action_repeat = 1
        log.debug('Using action repeat %d during evaluation',
                  render_action_repeat)

        cfg.env_frameskip = 1  # for evaluation
        cfg.num_envs = 1
        # this config is used for computing displacement errors
        ade_cfg = deepcopy(cfg)
        ade_cfg['remove_at_goal'] = False
        ade_cfg['remove_at_collide'] = False

        def make_env_func(env_config):
            return create_env(cfg.env, cfg=cfg, env_config=env_config)

        env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
        env.seed(0)

        is_multiagent = is_multiagent_env(env)
        if not is_multiagent:
            env = MultiAgentWrapper(env)

        if hasattr(env.unwrapped, 'reset_on_init'):
            # reset call ruins the demo recording for VizDoom
            env.unwrapped.reset_on_init = False

        actor_critic = create_actor_critic(cfg, env.observation_space,
                                           env.action_space)

        device = torch.device(device)
        actor_critic.model_to_device(device)

        policy_id = cfg.policy_index
        checkpoints = LearnerWorker.get_checkpoints(
            LearnerWorker.checkpoint_dir(cfg, policy_id))
        checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
        actor_critic.load_state_dict(checkpoint_dict['model'])
        actor_critics.append([i, actor_critic])

    # we bin the success rate into bins of 10 meters between 0 and 400
    # the second dimension is the counts
    distance_bins = np.linspace(0, 400, 40)
    intersections_bins = np.linspace(0, 7, 7)
    num_files = cfg['num_eval_files']
    num_file_loops = cfg['num_file_loops']
    # TODO(eugenevinitsky) horrifying copy and paste
    if test_zsc:
        goal_array = np.zeros((len(actor_critics), len(actor_critics),
                               num_file_loops * num_files))
        collision_array = np.zeros((len(actor_critics), len(actor_critics),
                                    num_files * num_file_loops))
        success_rate_by_num_agents = np.zeros(
            (len(actor_critics), len(actor_critics), cfg.max_num_vehicles, 4))
        success_rate_by_distance = np.zeros(
            (len(actor_critics), len(actor_critics), distance_bins.shape[-1],
             4))
        success_rate_by_intersections = np.zeros(
            (len(actor_critics), len(actor_critics),
             intersections_bins.shape[-1], 4))
        ade_array = np.zeros((len(actor_critics), len(actor_critics),
                              num_file_loops * num_files))
        fde_array = np.zeros((len(actor_critics), len(actor_critics),
                              num_file_loops * num_files))
        veh_veh_collision_array = np.zeros(
            (len(actor_critics), len(actor_critics),
             num_file_loops * num_files))
        veh_edge_collision_array = np.zeros(
            (len(actor_critics), len(actor_critics),
             num_file_loops * num_files))
    else:
        goal_array = np.zeros((len(actor_critics), num_file_loops * num_files))
        collision_array = np.zeros(
            (len(actor_critics), num_file_loops * num_files))
        veh_veh_collision_array = np.zeros(
            (len(actor_critics), num_file_loops * num_files))
        veh_edge_collision_array = np.zeros(
            (len(actor_critics), num_file_loops * num_files))
        success_rate_by_num_agents = np.zeros(
            (len(actor_critics), cfg.max_num_vehicles, 4))
        success_rate_by_distance = np.zeros(
            (len(actor_critics), distance_bins.shape[-1], 4))
        success_rate_by_intersections = np.zeros(
            (len(actor_critics), intersections_bins.shape[-1], 4))
        ade_array = np.zeros((len(actor_critics), num_file_loops * num_files))
        fde_array = np.zeros((len(actor_critics), num_file_loops * num_files))

    if test_zsc:
        output_generator = itertools.product(actor_critics, actor_critics)
    else:
        output_generator = actor_critics

    for output in output_generator:
        if test_zsc:
            (index_1, actor_1), (index_2, actor_2) = output
        else:
            (index_1, actor_1) = output
        goal_frac = []
        collision_frac = []
        veh_veh_collision_frac = []
        veh_edge_collision_frac = []
        average_displacement_error = []
        final_displacement_error = []
        veh_counter = 0
        for loop_num in range(num_file_loops):
            for file_num, file in enumerate(files[0:cfg['num_eval_files']]):
                print(loop_num * cfg['num_eval_files'] + file_num)
                print('file is {}'.format(os.path.join(scenario_dir, file)))

                env.unwrapped.files = [os.path.join(scenario_dir, file)]

                # step the env to its conclusion to generate the expert trajectories we compare against
                env.cfg = ade_cfg
                env.reset()
                expert_trajectory_dict = defaultdict(lambda: np.zeros((80, 2)))
                env.unwrapped.make_all_vehicles_experts()
                for i in range(80):
                    for veh in env.unwrapped.get_objects_that_moved():
                        expert_trajectory_dict[
                            veh.id][i] = veh.position.numpy()
                    env.unwrapped.simulation.step(0.1)

                # compute the number of expert trajectories that intersect
                # while filtering out the bits of the trajectory
                # that were invalid
                vehs_with_intersecting_ids = defaultdict(int)
                for veh_id in expert_trajectory_dict.keys():
                    for veh_id2 in expert_trajectory_dict.keys():
                        if veh_id == veh_id2:
                            continue
                        trajectory = expert_trajectory_dict[veh_id]
                        trajectory2 = expert_trajectory_dict[veh_id2]
                        expert_mask_arr = trajectory.sum(axis=1)
                        expert_mask = (expert_mask_arr != 0.0) * (
                            expert_mask_arr != trajectory.shape[1] * ERR_VAL)
                        trajectory = trajectory[expert_mask]
                        expert_mask_arr = trajectory2.sum(axis=1)
                        expert_mask = (expert_mask_arr != 0.0) * (
                            expert_mask_arr != trajectory2.shape[1] * ERR_VAL)
                        trajectory2 = trajectory2[expert_mask]
                        if poly_intersection(trajectory, trajectory2):
                            vehs_with_intersecting_ids[
                                veh_id] += poly_intersection(
                                    trajectory, trajectory2)

                env.cfg = cfg
                if test_zsc:
                    output = run_rollouts(env, cfg, device,
                                          expert_trajectory_dict,
                                          distance_bins, intersections_bins,
                                          vehs_with_intersecting_ids, actor_1,
                                          actor_2)
                else:
                    output = run_rollouts(env, cfg, device,
                                          expert_trajectory_dict,
                                          distance_bins, intersections_bins,
                                          vehs_with_intersecting_ids, actor_1)

                avg_goal, avg_collisions, avg_veh_edge_collisions, avg_veh_veh_collisions, \
                    success_rate_by_distance_return, success_rate_by_num_agents_return, \
                    success_rate_by_intersections_return, \
                    _, _, _ = output
                # TODO(eugenevinitsky) hideous copy and pasting
                goal_frac.append(avg_goal)
                collision_frac.append(avg_collisions)
                veh_veh_collision_frac.append(avg_veh_veh_collisions)
                veh_edge_collision_frac.append(avg_veh_edge_collisions)
                if test_zsc:
                    success_rate_by_distance[
                        index_1, index_2] += success_rate_by_distance_return
                    success_rate_by_num_agents[
                        index_1, index_2] += success_rate_by_num_agents_return
                    success_rate_by_intersections[
                        index_1,
                        index_2] += success_rate_by_intersections_return
                else:
                    success_rate_by_distance[
                        index_1] += success_rate_by_distance_return
                    success_rate_by_num_agents[
                        index_1] += success_rate_by_num_agents_return
                    success_rate_by_intersections[
                        index_1] += success_rate_by_intersections_return
                # do some logging
                log.info(
                    f'Avg goal achieved {np.mean(goal_frac)}±{np.std(goal_frac) / len(goal_frac)}'
                )
                log.info(
                    f'Avg veh-veh collisions {np.mean(veh_veh_collision_frac)}±\
                        {np.std(veh_veh_collision_frac) / np.sqrt(len(veh_veh_collision_frac))}'
                )
                log.info(
                    f'Avg veh-edge collisions {np.mean(veh_edge_collision_frac)}±\
                        {np.std(veh_edge_collision_frac) / np.sqrt(len(veh_edge_collision_frac))}'
                )
                log.info(f'Avg num collisions {np.mean(collision_frac)}±\
                        {np.std(collision_frac) / len(collision_frac)}')

                env.cfg = ade_cfg
                # okay, now run the rollout one more time but this time set
                # remove_at_goal and remove_at_collide to be false so we can do the ADE computations
                if test_zsc:
                    output = run_rollouts(env, cfg, device,
                                          expert_trajectory_dict,
                                          distance_bins, intersections_bins,
                                          vehs_with_intersecting_ids, actor_1,
                                          actor_2)
                else:
                    output = run_rollouts(env, cfg, device,
                                          expert_trajectory_dict,
                                          distance_bins, intersections_bins,
                                          vehs_with_intersecting_ids, actor_1)

                _, _, _, _, _, _, _, ade, fde, veh_counter = output
                average_displacement_error.append(ade)
                final_displacement_error.append(fde)
                log.info(f'Avg ADE {np.mean(average_displacement_error)}±\
                        {np.std(average_displacement_error) / np.sqrt(len(average_displacement_error))}'
                         )
                log.info(f'Avg FDE {np.mean(final_displacement_error)}±\
                        {np.std(final_displacement_error) / np.sqrt(len(final_displacement_error))}'
                         )

        if test_zsc:
            goal_array[index_1, index_2] = goal_frac
            collision_array[index_1, index_2] = collision_frac
            veh_veh_collision_array[index_1, index_2] = veh_veh_collision_frac
            veh_edge_collision_array[index_1,
                                     index_2] = veh_edge_collision_frac
            ade_array[index_1, index_2] = average_displacement_error
            fde_array[index_1, index_2] = final_displacement_error
        else:
            goal_array[index_1] = goal_frac
            collision_array[index_1] = collision_frac
            veh_veh_collision_array[index_1] = veh_veh_collision_frac
            veh_edge_collision_array[index_1] = veh_edge_collision_frac
            ade_array[index_1] = average_displacement_error
            fde_array[index_1] = final_displacement_error

    if test_zsc:
        file_type += '_zsc'
    np.save(os.path.join(output_path, '{}_goal.npy'.format(file_type)),
            goal_array)
    np.save(os.path.join(output_path, '{}_collision.npy'.format(file_type)),
            collision_array)
    np.save(
        os.path.join(output_path,
                     '{}_veh_veh_collision.npy'.format(file_type)),
        veh_veh_collision_array)
    np.save(
        os.path.join(output_path,
                     '{}_veh_edge_collision.npy'.format(file_type)),
        veh_edge_collision_array)
    np.save(os.path.join(output_path, '{}_ade.npy'.format(file_type)),
            ade_array)
    np.save(os.path.join(output_path, '{}_fde.npy'.format(file_type)),
            fde_array)
    with open(
            os.path.join(output_path,
                         '{}_success_by_veh_number.npy'.format(file_type)),
            'wb') as f:
        np.save(f, success_rate_by_num_agents)
    with open(
            os.path.join(output_path,
                         '{}_success_by_dist.npy'.format(file_type)),
            'wb') as f:
        np.save(f, success_rate_by_distance)
    with open(
            os.path.join(
                output_path,
                '{}_success_by_num_intersections.npy'.format(file_type)),
            'wb') as f:
        np.save(f, success_rate_by_intersections)

    env.close()

    return


def load_wandb(experiment_name, cfg_filter, force_reload=False):
    """Pull the results from the wandb server.

    Args:
    ----
        experiment_name (str): name of the wandb group.
        cfg_filter (function): use the config dict to filter
                               which runs are actually kept
        force_reload (bool, optional): if true we overwrite
                                       the wandb csv
                                       even if it exists.
    """
    if not os.path.exists(
            'wandb_{}.csv'.format(experiment_name)) or force_reload:
        import wandb

        api = wandb.Api()
        entity, project = "eugenevinitsky", "nocturne4"  # set to your entity and project
        runs = api.runs(entity + "/" + project)

        history_list = []
        for run in runs:
            if run.name == experiment_name:

                # # .config contains the hyperparameters.
                # #  We remove special values that start with _.
                config = {
                    k: v
                    for k, v in run.config.items() if not k.startswith('_')
                }
                if cfg_filter(config):
                    history_df = run.history()
                    history_df['seed'] = config['seed']
                    history_df['num_files'] = config['num_files']
                    history_list.append(history_df)

        runs_df = pd.concat(history_list)
        runs_df.to_csv('wandb_{}.csv'.format(experiment_name))


def plot_goal_achieved(experiment_name, global_step_cutoff=3e9):
    """Use the WANDB CSV to plot number of train steps v. goal achieved."""
    plt.figure(dpi=300)
    df = pd.read_csv("wandb_{}.csv".format(experiment_name))
    df["timestamp"] = pd.to_datetime(df["_timestamp"] * 1e9)

    # technically not correct if the number of seeds varies by num_files
    # but in this case we're alright
    num_seeds = len(np.unique(df.seed.values))

    values_num_files = np.unique(df.num_files.values)
    column = "0_aux/avg_goal_achieved"
    dfs = []
    stdevs = []
    for num_files in values_num_files:
        if num_files == 1:
            continue

        df_n = df[(df.num_files == num_files)
                  & (df.global_step < global_step_cutoff)].set_index(
                      'global_step').sort_index()
        if num_files == -1:
            col_name = 134453
        else:
            col_name = num_files
        dfs.append((df_n[column] * 100).ewm(
            halflife=500,
            min_periods=10).mean().rename(f"num_files={col_name}"))
        stdevs.append((df_n[column] * 100).ewm(halflife=500,
                                               min_periods=10).std())

    values_num_files = [
        val if val != -1 else 134453 for val in values_num_files
    ]
    temp = list(zip(values_num_files, dfs, stdevs))
    temp = sorted(temp, key=lambda x: x[0])
    values_num_files, dfs, stdevs = zip(*temp)
    ax = plt.gca()
    for i in range(len(dfs)):
        x = dfs[i].index.values
        y = dfs[i].values
        yerr = stdevs[i].replace(np.nan, 0) / np.sqrt(num_seeds)
        ax.plot(x,
                y,
                label=f'Training Files: {values_num_files[i]}',
                color=CB_color_cycle[i])
        ax.fill_between(x,
                        y - 2 * yerr,
                        y + 2 * yerr,
                        color=CB_color_cycle[i],
                        alpha=0.3)
    plt.grid(ls='--', color='#ccc')
    plt.legend()
    plt.xlabel("Environment step")
    plt.ylabel("% Goals Achieved")
    plt.savefig('goal_achieved_v_step', bbox_inches='tight', pad_inches=0.1)


def eval_generalization(output_folder,
                        num_eval_files,
                        files,
                        file_type,
                        scenario_dir,
                        num_file_loops,
                        test_zsc=False,
                        cfg_filter=None):
    """Evaluate generalization for all agent checkpoints in output_folder.

    Args:
    ----
        output_folder (str): path to folder containing agent checkpoints
        num_eval_files (int): how many files to use for eval
        files (list[str]): list of scenario files to use for eval
        file_type (str): 'train' or 'test' used to indicate if we are
                         testing in or out of distribution
        scenario_dir (str): path to directory where `files` are stored
        num_file_loops (int): how many times to iterate over the files.
                              Used for in-distribution testing if
                              in-distribution we trained on M files
                              but we want to test over N files where
                              N > M.
        test_zsc (bool, optional): If true we pair up ever
                                   agent in the folder and compute
                                   all the cross-play scores. Defaults to False.
        cfg_filter (_type_, optional): function used to filter over
                                       whether eval should actually be done on that
                                       agent. Filters using the agent config dict.
    """
    file_paths = []
    cfg_dicts = []
    for (dirpath, dirnames, filenames) in os.walk(output_folder):
        if 'cfg.json' in filenames:
            with open(os.path.join(dirpath, 'cfg.json'), 'r') as file:
                cfg_dict = json.load(file)

            if cfg_filter is not None and not cfg_filter(cfg_dict):
                continue
            file_paths.append(dirpath)
            cfg_dict['cli_args'] = {}
            cfg_dict['fps'] = 0
            cfg_dict['render_action_repeat'] = None
            cfg_dict['no_render'] = None
            cfg_dict['policy_index'] = 0
            cfg_dict['record_to'] = os.path.join(os.getcwd(), '..', 'recs')
            cfg_dict['continuous_actions_sample'] = False
            cfg_dict['discrete_actions_sample'] = False
            # for the train set, we don't want to loop over
            # files we didn't train on
            # also watch out for -1 which means "train on all files"
            if cfg_dict[
                    'num_files'] < num_eval_files and 'train' in file_type and cfg_dict[
                        'num_files'] != -1:
                cfg_dict['num_eval_files'] = cfg_dict['num_files']
                cfg_dict['num_file_loops'] = num_file_loops * int(
                    max(num_eval_files // cfg_dict['num_files'], 1))
            else:
                cfg_dict['num_eval_files'] = num_eval_files
                cfg_dict['num_file_loops'] = num_file_loops
            cfg_dicts.append(cfg_dict)
    if test_zsc:
        # TODO(eugenevinitsky) we're currently storing the ZSC result in a random
        # folder which seems bad.
        run_eval([Bunch(cfg_dict) for cfg_dict in cfg_dicts],
                 test_zsc=test_zsc,
                 output_path=file_paths[0],
                 scenario_dir=scenario_dir,
                 files=files,
                 file_type=file_type)
        print('stored ZSC result in {}'.format(file_paths[0]))
    else:
        # why 13? because a 16 GB GPU can do a forwards pass on 13 copies of the model
        # for 20 vehicles at once. More than that and you'll run out of memory
        num_cpus = min(13, mp.cpu_count() - 2)
        device = 'cuda'
        # if torch.cuda.is_available():
        #     device = 'cuda'
        # else:
        #     device = 'cpu'
        with mp.Pool(processes=num_cpus) as pool:
            list(
                pool.starmap(
                    run_eval,
                    zip(cfg_dicts, repeat(test_zsc), file_paths,
                        repeat(scenario_dir), repeat(files), repeat(file_type),
                        repeat(device))))
    print(file_paths)


def main():
    """Script entry point."""
    set_display_window()
    register_custom_components()
    RUN_EVAL = True
    TEST_ZSC = False
    PLOT_RESULTS = True
    RELOAD_WANDB = False
    VERSION = 5
    NUM_EVAL_FILES = 200
    NUM_FILE_LOOPS = 1  # the number of times to loop over a fixed set of files
    # experiment_names = ['srt_v27']
    experiment_names = [<THE NAME YOU GAVE THE EXPERIMENT WHEN YOU RAN IT>]
    output_folder = [<PATH TO THE OUTER FOLDER WHERE 0, 1, 2, 3, 4 are stored>]
    # output_folder = '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.20/new_road_sample/18.32.35'
    # output_folder = [
    #     '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.23/srt_v10/17.02.40/'
    # ]
    # 10 files
    # output_folder = [
    #     '/checkpoint/eugenevinitsky/nocturne/sweep/2022.05.28/srt_12/16.43.16/'
    # ]
    # SRT submission results
    # output_folder = [
    #     '/checkpoint/eugenevinitsky/nocturne/sweep/2022.06.01/srt_v27/17.35.33'
    # ]
    generalization_dfs = []

    cfg_filter = None

    if TEST_ZSC:

        def cfg_filter(cfg_dict):
            if cfg_dict['scenario']['road_edge_first'] is False and cfg_dict[
                    'scenario']['max_visible_road_points'] == 500 and cfg_dict[
                        'algorithm']['encoder_hidden_size'] == 256 and cfg_dict[
                            'num_files'] == 10000:
                return True
            else:
                return False
    else:

        def cfg_filter(cfg_dict):
            if cfg_dict['scenario']['road_edge_first'] is False and cfg_dict[
                    'scenario']['max_visible_road_points'] == 500 and cfg_dict[
                        'algorithm']['encoder_hidden_size'] == 256:
                return True
            else:
                return False

    '''
    ###############################################################################
    #########           Build the generalization dataframes ######################
    ##############################################################################
    '''

    if RUN_EVAL:
        if TEST_ZSC:
            output_generator = [(PROCESSED_VALID_NO_TL,
                                 'test_{}'.format(VERSION))]
        else:
            output_generator = [
                (PROCESSED_TRAIN_NO_TL, 'train_{}'.format(VERSION)),
                (PROCESSED_VALID_NO_TL, 'test_{}'.format(VERSION))
            ]

        for file_path, file_type in output_generator:
            with open(os.path.join(file_path, 'valid_files.json')) as file:
                valid_veh_dict = json.load(file)
                files = list(valid_veh_dict.keys())
                if file_type == 'test_{}'.format(VERSION):
                    # sort the files so that we have a consistent order
                    np.random.seed(0)
                    np.random.shuffle(files)
                if file_type == 'train_{}'.format(VERSION):
                    # for train make sure we use the same ordering
                    # that is used in base_env
                    # TODO(eugenevinitsky) this is dangerous and could
                    # break easily
                    files = sorted(files)
            for folder in output_folder:
                eval_generalization(folder,
                                    NUM_EVAL_FILES,
                                    files,
                                    file_type=file_type,
                                    scenario_dir=file_path,
                                    num_file_loops=NUM_FILE_LOOPS,
                                    test_zsc=TEST_ZSC,
                                    cfg_filter=cfg_filter)

    if PLOT_RESULTS:
        # okay, now build a pandas dataframe of the results that we will use for plotting
        # the generalization results
        for folder in output_folder:
            for file_type in [
                    'train_{}'.format(VERSION), 'test_{}'.format(VERSION)
                    # 'train',
                    # 'test'
            ]:
                file_paths = []
                data_dicts = []
                for (dirpath, dirnames, filenames) in os.walk(folder):
                    if 'cfg.json' in filenames:
                        file_paths.append(dirpath)
                        with open(os.path.join(dirpath, 'cfg.json'),
                                  'r') as file:
                            cfg_dict = json.load(file)
                        if cfg_filter(cfg_dict):
                            # TODO(eugenevinitsky) why do they not all have this?
                            goal = np.mean(
                                np.load(
                                    os.path.join(
                                        dirpath,
                                        '{}_goal.npy'.format(file_type))))
                            collide = np.mean(
                                np.load(
                                    os.path.join(
                                        dirpath,
                                        '{}_collision.npy'.format(file_type))))
                            ade = np.mean(
                                np.load(
                                    os.path.join(
                                        dirpath,
                                        '{}_ade.npy'.format(file_type))))
                            fde = np.mean(
                                np.load(
                                    os.path.join(
                                        dirpath,
                                        '{}_fde.npy'.format(file_type))))
                            veh_veh_collision = np.mean(
                                np.load(
                                    os.path.join(
                                        dirpath,
                                        '{}_veh_veh_collision.npy'.format(
                                            file_type))))
                            veh_edge_collision = np.mean(
                                np.load(
                                    os.path.join(
                                        dirpath,
                                        '{}_veh_edge_collision.npy'.format(
                                            file_type))))
                            success_by_num_intersections = np.load(
                                os.path.join(
                                    dirpath,
                                    '{}_success_by_num_intersections.npy'.
                                    format(file_type)))
                            # there aren't a lot of data points past 3
                            # so just bundle them in
                            success_by_num_intersections[:,
                                                         3, :] = success_by_num_intersections[:, 3:, :].sum(
                                                             axis=1)
                            success_by_num_intersections = success_by_num_intersections[:,
                                                                                        0:
                                                                                        4, :]
                            success_by_veh_num = np.load(
                                os.path.join(
                                    dirpath,
                                    '{}_success_by_veh_number.npy'.format(
                                        file_type)))
                            success_by_distance = np.load(
                                os.path.join(
                                    dirpath, '{}_success_by_dist.npy'.format(
                                        file_type)))
                            num_files = cfg_dict['num_files']
                            if int(num_files) == -1:
                                num_files = 134453
                            if int(num_files) == 1:
                                continue
                            data_dicts.append({
                                'num_files':
                                num_files,
                                'goal_rate':
                                goal * 100,
                                'collide_rate':
                                collide * 100,
                                'ade':
                                ade,
                                'fde':
                                fde,
                                'veh_veh_collision':
                                veh_veh_collision,
                                'veh_edge_collision':
                                veh_edge_collision,
                                'goal_by_intersections':
                                np.nan_to_num(
                                    success_by_num_intersections[0, :, 0] /
                                    success_by_num_intersections[0, :, 3]),
                                'collide_by_intersections':
                                np.nan_to_num(
                                    success_by_num_intersections[0, :, 1] /
                                    success_by_num_intersections[0, :, 3]),
                                'goal_by_vehicle_num':
                                np.nan_to_num(success_by_veh_num[0, :, 0] /
                                              success_by_veh_num[0, :, 3]),
                                'collide_by_vehicle_num':
                                np.nan_to_num(success_by_veh_num[0, :, 1] /
                                              success_by_veh_num[0, :, 3]),
                                'goal_by_distance':
                                np.nan_to_num(success_by_distance[0, :, 0] /
                                              success_by_distance[0, :, 3]),
                                'collide_by_distance':
                                np.nan_to_num(success_by_distance[0, :, 1] /
                                              success_by_distance[0, :, 3]),
                            })
                            if cfg_dict['num_files'] == 10000:
                                print('goal ',
                                      success_by_num_intersections[0, :, 0])
                                print('num vehicles in bin',
                                      success_by_num_intersections[0, :, 3])
                df = pd.DataFrame(data_dicts)
                new_dict = {}
                for key in data_dicts[0].keys():
                    if key == 'num_files':
                        continue
                    new_dict[key] = df.groupby(['num_files'
                                                ])[key].mean().reset_index()
                    try:
                        new_dict[key + '_std'] = df.groupby(
                            ['num_files'])[key].std().reset_index().rename(
                                columns={key: key + '_std'})
                    except ValueError:
                        # TODO(eugenevinitsky) learn to use pandas dawg
                        # what even is this
                        temp_dict = {}
                        for name, group in df.groupby(['num_files'])[key]:
                            temp = []
                            for arr in group:
                                temp.append(arr)
                            np_arr = np.vstack(temp)
                            std_err = np.std(np_arr, axis=0) / np.sqrt(
                                np_arr.shape[0])
                            temp_dict[name] = std_err
                        new_dict[key + '_stderr'] = pd.Series(
                            data=temp_dict).reset_index().rename(
                                columns={
                                    'index': 'num_files',
                                    0: key + '_stderr'
                                })
                first_elem_key = 'goal_rate'
                first_elem = new_dict[first_elem_key]
                for key, value in new_dict.items():
                    if key == first_elem_key:
                        continue
                    first_elem = first_elem.merge(value,
                                                  how='inner',
                                                  on='num_files')
                generalization_dfs.append(first_elem)
            '''
        ###############################################################################
        #########  load the training dataframes from wandb ######################
        ##############################################################################
        '''
        # global_step_cutoff = 3e9
        # training_dfs = []
        # for experiment_name in experiment_names:
        #     load_wandb(experiment_name, cfg_filter, force_reload=RELOAD_WANDB)
        #     training_dfs.append(
        #         pd.read_csv('wandb_{}.csv'.format(experiment_name)))

        num_seeds = 5 #len(np.unique(training_dfs[0].seed))
        # create the goal plot
        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            plt.plot(np.log10(df.num_files),
                     df.goal_rate,
                     color=CB_color_cycle[i],
                     label=file_type)
            ax = plt.gca()
            yerr = df.goal_rate_std.replace(np.nan, 0) / np.sqrt(num_seeds)
            ax.fill_between(np.log10(df.num_files),
                            df.goal_rate - 2 * yerr,
                            df.goal_rate + 2 * yerr,
                            color=CB_color_cycle[i],
                            alpha=0.3)
            print(f'{file_type} goal rate', df.goal_rate, yerr)
        plt.ylim([0, 100])
        plt.xlabel(' Number of Training Files (Logarithmic Scale)')
        plt.ylabel('% Goals Achieved')
        plt.legend()
        plt.savefig('goal_achieved.png', bbox_inches='tight', pad_inches=0.1)

        # create the collide plot
        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            plt.plot(np.log10(df.num_files),
                     df.collide_rate,
                     color=CB_color_cycle[i],
                     label=file_type)
            ax = plt.gca()
            yerr = df.collide_rate_std.replace(np.nan, 0) / np.sqrt(num_seeds)
            ax.fill_between(np.log10(df.num_files),
                            df.collide_rate - 2 * yerr,
                            df.collide_rate + 2 * yerr,
                            color=CB_color_cycle[i],
                            alpha=0.3)
            print(f'{file_type} collide rate', df.collide_rate, yerr)
        plt.ylim([0, 50])
        plt.xlabel(' Number of Training Files (Logarithmic Scale)')
        plt.ylabel('% Vehicles Collided')
        plt.legend()
        plt.savefig('collide_rate.png', bbox_inches='tight', pad_inches=0.1)

        # create ADE and FDE plots

        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            yerr = df.ade_std.replace(np.nan, 0) / np.sqrt(num_seeds)
            plt.plot(np.log10(df.num_files),
                     df.ade,
                     label=file_type,
                     color=CB_color_cycle[i])
            ax = plt.gca()
            ax.fill_between(np.log10(df.num_files),
                            df.ade - 2 * yerr,
                            df.ade + 2 * yerr,
                            color=CB_color_cycle[i],
                            alpha=0.3)
            print(f'{file_type} ade', df.ade, yerr)
        plt.xlabel(' Number of Training Files (Logarithmic Scale)')
        plt.ylabel('Average Displacement Error (m)')
        plt.ylim([0, 5])
        plt.legend()
        plt.savefig('ade.png', bbox_inches='tight', pad_inches=0.1)

        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            yerr = df.fde_std.replace(np.nan, 0) / np.sqrt(num_seeds)
            plt.plot(np.log10(df.num_files),
                     df.fde,
                     label=file_type,
                     color=CB_color_cycle[i])
            ax = plt.gca()
            ax.fill_between(np.log10(df.num_files),
                            df.fde - 2 * yerr,
                            df.fde + 2 * yerr,
                            color=CB_color_cycle[i],
                            alpha=0.3)
            print(f'{file_type} fde', df.fde, yerr)
        plt.ylim([4, 10])
        plt.xlabel(' Number of Training Files (Logarithmic Scale)')
        plt.ylabel('Final Displacement Error (m)')
        plt.legend()
        plt.savefig('fde.png', bbox_inches='tight', pad_inches=0.1)
        plot_goal_achieved(experiment_names[0], global_step_cutoff)

        # create error by number of expert intersections plots
        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            values_num_files = np.unique(df.num_files.values)
            print(values_num_files)
            for value in values_num_files:
                if value != 10000:
                    continue
                numpy_arr = df[df.num_files ==
                               value]['goal_by_intersections'].to_numpy()[0]
                temp_df = pd.DataFrame(numpy_arr).melt()
                plt.plot(temp_df.index,
                         temp_df.value * 100,
                         label=file_type,
                         color=CB_color_cycle[i])
                numpy_arr = df[df.num_files == value][
                    'goal_by_intersections_stderr'].to_numpy()[0]
                std_err_df = pd.DataFrame(numpy_arr).melt()
                ax = plt.gca()
                ax.fill_between(temp_df.index,
                                100 * (temp_df.value - 2 * std_err_df.value),
                                100 * (temp_df.value + 2 * std_err_df.value),
                                color=CB_color_cycle[i],
                                alpha=0.3)

        plt.xlabel('Number of intersecting paths')
        plt.ylabel('Percent Goals Achieved')
        ax.set_xticks([i for i in range(numpy_arr.shape[-1])])
        plt.legend()
        plt.savefig('goal_v_intersection.png',
                    bbox_inches='tight',
                    pad_inches=0.1)

        # create error by number of expert intersections plots
        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            values_num_files = np.unique(df.num_files.values)
            for value in values_num_files:
                if value != 10000:
                    continue
                numpy_arr = df[df.num_files ==
                               value]['collide_by_intersections'].to_numpy()[0]
                temp_df = pd.DataFrame(numpy_arr).melt()
                plt.plot(temp_df.index,
                         temp_df.value * 100,
                         color=CB_color_cycle[i],
                         label=file_type)
                numpy_arr = df[df.num_files == value][
                    'collide_by_intersections_stderr'].to_numpy()[0]
                std_err_df = pd.DataFrame(numpy_arr).melt()
                ax = plt.gca()
                ax.fill_between(temp_df.index,
                                100 * (temp_df.value - 2 * std_err_df.value),
                                100 * (temp_df.value + 2 * std_err_df.value),
                                color=CB_color_cycle[i],
                                alpha=0.3)
        plt.xlabel('Number of Intersecting Paths')
        plt.ylabel('Percent Collisions')
        ax.set_xticks([i for i in range(numpy_arr.shape[-1])])
        plt.legend()
        plt.savefig('collide_v_intersection.png',
                    bbox_inches='tight',
                    pad_inches=0.1)

        # create error by number of vehicles plots
        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            values_num_files = np.unique(df.num_files.values)
            print(values_num_files)
            for value in values_num_files:
                if value != 10000:
                    continue
                numpy_arr = df[df.num_files ==
                               value]['goal_by_vehicle_num'].to_numpy()[0]
                temp_df = pd.DataFrame(numpy_arr).melt()
                plt.plot(temp_df.index,
                         temp_df.value * 100,
                         label=file_type,
                         color=CB_color_cycle[i])
                numpy_arr = df[df.num_files == value][
                    'goal_by_vehicle_num_stderr'].to_numpy()[0]
                std_err_df = pd.DataFrame(numpy_arr).melt()
                ax = plt.gca()
                ax.fill_between(temp_df.index,
                                100 * (temp_df.value - 2 * std_err_df.value),
                                100 * (temp_df.value + 2 * std_err_df.value),
                                color=CB_color_cycle[i],
                                alpha=0.3)
                # sns.lineplot(x=temp_df.index, y=temp_df.value * 100)
        plt.xlabel('Number of Controlled Vehicles')
        plt.ylabel('Percent Goals Achieved')
        ax.set_xticks([i for i in range(numpy_arr.shape[-1])])
        plt.legend()
        plt.savefig('goal_v_vehicle_num.png',
                    bbox_inches='tight',
                    pad_inches=0.1)

        # create error by distance plots
        plt.figure(dpi=300)
        for i, (df, file_type) in enumerate(
                zip(generalization_dfs, ['Train', 'Test'])):
            values_num_files = np.unique(df.num_files.values)
            print(values_num_files)
            for value in values_num_files:
                if value != 10000:
                    continue
                numpy_arr = df[df.num_files ==
                               value]['goal_by_distance'].to_numpy()[0]
                temp_df = pd.DataFrame(numpy_arr).melt()
                plt.plot(temp_df.index,
                         temp_df.value * 100,
                         label=file_type,
                         color=CB_color_cycle[i])
                numpy_arr = df[df.num_files ==
                               value]['goal_by_distance_stderr'].to_numpy()[0]
                std_err_df = pd.DataFrame(numpy_arr).melt()
                ax = plt.gca()
                ax.fill_between(temp_df.index,
                                100 * (temp_df.value - 2 * std_err_df.value),
                                100 * (temp_df.value + 2 * std_err_df.value),
                                color=CB_color_cycle[i],
                                alpha=0.3)
                # sns.lineplot(x=temp_df.index, y=temp_df.value * 100)
        plt.xlabel('Starting Distance to Goal')
        plt.ylabel('Percent Goals Achieved')
        ax.set_xticks([i for i in range(numpy_arr.shape[-1])])
        plt.legend()
        plt.savefig('goal_v_distance.png', bbox_inches='tight', pad_inches=0.1)


if __name__ == '__main__':
    sys.exit(main())
