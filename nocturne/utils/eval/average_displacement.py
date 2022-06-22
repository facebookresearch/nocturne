# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Average displacement error computation."""
from collections import defaultdict
from itertools import repeat
import json
from multiprocessing import Pool
import os
import random

import numpy as np
import torch

from cfgs.config import PROCESSED_VALID_NO_TL, ERR_VAL
from nocturne import Simulation

SIM_N_STEPS = 90  # number of steps per trajectory
GOAL_TOLERANCE = 0.5


def _average_displacement_impl(arg):
    trajectory_path, model, configs = arg
    print(trajectory_path)

    scenario_config = configs['scenario_cfg']

    view_dist = configs['dataloader_cfg']['view_dist']
    view_angle = configs['dataloader_cfg']['view_angle']
    state_normalization = configs['dataloader_cfg']['state_normalization']
    dt = configs['dataloader_cfg']['dt']

    n_stacked_states = configs['dataloader_cfg']['n_stacked_states']
    state_size = configs['model_cfg']['n_inputs'] // n_stacked_states
    state_dict = defaultdict(lambda: np.zeros(state_size * n_stacked_states))

    # create expert simulation
    sim_expert = Simulation(str(trajectory_path), scenario_config)
    scenario_expert = sim_expert.getScenario()
    vehicles_expert = scenario_expert.getVehicles()
    objects_expert = scenario_expert.getObjectsThatMoved()
    id2veh_expert = {veh.id: veh for veh in vehicles_expert}

    # create model simulation
    sim_model = Simulation(str(trajectory_path), scenario_config)
    scenario_model = sim_model.getScenario()
    vehicles_model = scenario_model.getVehicles()
    objects_model = scenario_model.getObjectsThatMoved()

    # set all objects to be expert-controlled
    for obj in objects_expert:
        obj.expert_control = True
    for obj in objects_model:
        obj.expert_control = True

    # in model sim, model will control vehicles that moved
    controlled_vehicles = [
        veh for veh in vehicles_model if veh in objects_model
    ]
    random.shuffle(controlled_vehicles)
    # controlled_vehicles = controlled_vehicles[:2]

    # warmup to build up state stacking
    for i in range(n_stacked_states - 1):
        for veh in controlled_vehicles:
            ego_state = scenario_model.ego_state(veh)
            visible_state = scenario_model.flattened_visible_state(
                veh, view_dist=view_dist, view_angle=view_angle)
            state = np.concatenate(
                (ego_state, visible_state)) / state_normalization
            state_dict[veh.getID()] = np.roll(state_dict[veh.getID()],
                                              len(state))
            state_dict[veh.getID()][:len(state)] = state
        sim_model.step(dt)
        sim_expert.step(dt)

    for veh in controlled_vehicles:
        veh.expert_control = False

    avg_displacements = []
    final_displacements = [0 for _ in controlled_vehicles]
    collisions = [False for _ in controlled_vehicles]
    goal_achieved = [False for _ in controlled_vehicles]
    for i in range(SIM_N_STEPS - n_stacked_states):
        for veh in controlled_vehicles:
            if np.isclose(veh.position.x, ERR_VAL):
                veh.expert_control = True
            else:
                veh.expert_control = False
        # set model actions
        all_states = []
        for veh in controlled_vehicles:
            # get vehicle state
            state = np.concatenate(
                (scenario_model.ego_state(veh),
                 scenario_model.flattened_visible_state(
                     veh, view_dist=view_dist,
                     view_angle=view_angle))) / state_normalization
            # stack state
            state_dict[veh.getID()] = np.roll(state_dict[veh.getID()],
                                              len(state))
            state_dict[veh.getID()][:len(state)] = state
            all_states.append(state_dict[veh.getID()])
        all_states = torch.as_tensor(np.array(all_states), dtype=torch.float32)

        # compute vehicle actions
        all_actions = model(all_states, deterministic=True
                            )  # /!\ this returns an array (2,n) and not (n,2)
        accel_actions = all_actions[0].cpu().numpy()
        steering_actions = all_actions[1].cpu().numpy()
        # set vehicles actions
        for veh, accel_action, steering_action in zip(controlled_vehicles,
                                                      accel_actions,
                                                      steering_actions):
            veh.acceleration = accel_action
            veh.steering = steering_action

        # step simulations
        sim_expert.step(dt)
        sim_model.step(dt)

        # compute displacements over non-collided vehicles
        displacements = []
        for i, veh in enumerate(controlled_vehicles):
            # get corresponding vehicle in expert simulation
            expert_veh = id2veh_expert[veh.id]
            # make sure it is valid
            if np.isclose(expert_veh.position.x,
                          ERR_VAL) or expert_veh.collided:
                continue
            # print(expert_veh.position, veh.position)
            # compute displacement
            expert_pos = id2veh_expert[veh.id].position
            model_pos = veh.position
            pos_diff = (model_pos - expert_pos).norm()
            displacements.append(pos_diff)
            final_displacements[i] = pos_diff
            if veh.collided:
                collisions[i] = True
            if (veh.position - veh.target_position).norm() < GOAL_TOLERANCE:
                goal_achieved[i] = True

        # average displacements over all vehicles
        if len(displacements) > 0:
            avg_displacements.append(np.mean(displacements))
            # print(displacements, np.mean(displacements))

    # average displacements over all time steps
    avg_displacement = np.mean(
        avg_displacements) if len(avg_displacements) > 0 else np.nan
    final_displacement = np.mean(
        final_displacements) if len(final_displacements) > 0 else np.nan
    avg_collisions = np.mean(collisions) if len(collisions) > 0 else np.nan
    avg_goals = np.mean(goal_achieved) if len(goal_achieved) > 0 else np.nan
    print('displacements', avg_displacement)
    print('final_displacement', final_displacement)
    print('collisions', avg_collisions)
    print('goal_rate', avg_goals)
    return avg_displacement, final_displacement, avg_collisions, avg_goals


def compute_average_displacement(trajectories_dir, model, configs):
    """Compute average displacement error between a model and the ground truth."""
    NUM_FILES = 200
    # get trajectories paths
    with open(os.path.join(trajectories_dir, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
        # sort the files so that we have a consistent order
        np.random.seed(0)
        np.random.shuffle(files)
    # compute average displacement over each individual trajectory file
    trajectories_paths = files[:NUM_FILES]
    for i, trajectory in enumerate(trajectories_paths):
        trajectories_paths[i] = os.path.join(trajectories_dir, trajectory)
    with Pool(processes=14) as pool:
        result = list(
            pool.map(_average_displacement_impl,
                     zip(trajectories_paths, repeat(model), repeat(configs))))
        average_displacements = np.array(result)[:, 0]
        final_displacements = np.array(result)[:, 1]
        average_collisions = np.array(result)[:, 2]
        average_goals = np.array(result)[:, 3]
        print(average_displacements, final_displacements, average_collisions,
              average_goals)

    return [
        np.mean(average_displacements[~np.isnan(average_displacements)]),
        np.std(average_displacements[~np.isnan(average_displacements)])
    ], [
        np.mean(final_displacements[~np.isnan(final_displacements)]),
        np.std(final_displacements[~np.isnan(final_displacements)])
    ], [
        np.mean(average_collisions[~np.isnan(average_collisions)]),
        np.std(average_collisions[~np.isnan(average_displacements)])
    ], [
        np.mean(average_goals[~np.isnan(average_goals)]),
        np.std(average_goals[~np.isnan(average_goals)])
    ]


if __name__ == '__main__':
    from examples.imitation_learning.model import ImitationAgent  # noqa: F401
    model = torch.load(
        '/checkpoint/eugenevinitsky/nocturne/test/2022.06.05/test/14.23.17/\
            ++device=cuda,++file_limit=1000/train_logs/2022_06_05_14_23_23/model_600.pth'
    ).to('cpu')
    model.actions_grids = [x.to('cpu') for x in model.actions_grids]
    model.eval()
    model.nn[0].eval()
    with open(
            '/checkpoint/eugenevinitsky/nocturne/test/2022.06.05/test/14.23.17/\
                ++device=cuda,++file_limit=1000/train_logs/2022_06_05_14_23_23/configs.json',
            'r') as fp:
        configs = json.load(fp)
        configs['device'] = 'cpu'
    with torch.no_grad():
        ade, fde, collisions, goals = compute_average_displacement(
            PROCESSED_VALID_NO_TL, model=model, configs=configs)
    print(f'Average Displacement Error: {ade[0]:.3f} ± {ade[1]:.3f} meters')
    print(f'Final Displacement Error: {fde[0]:.3f} ± {fde[1]:.3f} meters')
    print(f'Average Collisions: {collisions[0]:.3f} ± {collisions[1]:.3f}%')
    print(
        f'Average Success at getting to goal: {goals[0]:.3f} ± {goals[1]:.3f}%'
    )
