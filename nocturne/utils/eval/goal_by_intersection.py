# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Goal reaching rate and collision rate computation as a function of number of intersections in expert trajectory."""
from pathlib import Path
import numpy as np
import torch
from collections import defaultdict
import random
import json

from nocturne import Simulation
from cfgs.config import ERR_VAL as INVALID_POSITION
from multiprocessing import Pool
from itertools import repeat, combinations

SIM_N_STEPS = 90  # number of steps per trajectory
GOAL_TOLERANCE = 0.5


def _compute_expert_intersections(trajectory_path):
    with open(trajectory_path, 'r') as fp:
        data = json.load(fp)

    segments = defaultdict(list)
    for veh_id, veh in enumerate(data['objects']):
        # note: i checked and veh_id is consistent with how it's loaded in simulation

        for i in range(len(veh['position']) - 1):
            # compute polyline (might not be continuous since we have invalid positions)
            segment = np.array([
                [veh['position'][i]['x'], veh['position'][i]['y']],
                [veh['position'][i + 1]['x'], veh['position'][i + 1]['y']],
            ])

            # if segment doesnt contain an invalid position, append to trajectory
            if np.isclose(segment, INVALID_POSITION).any():
                continue
            segments[veh_id].append(segment)

    # go over pair of vehicles and check if their segments intersect
    n_collisions = defaultdict(int)
    for veh1, veh2 in combinations(segments.keys(), 2):
        # get corresponding segments
        segments1 = np.array(segments[veh1])
        segments2 = np.array(segments[veh2])

        # check bounding rectangle intersection - O(n)
        xmin1, ymin1 = np.min(np.min(segments1, axis=0), axis=0)
        xmax1, ymax1 = np.max(np.max(segments1, axis=0), axis=0)
        xmin2, ymin2 = np.min(np.min(segments2, axis=0), axis=0)
        xmax2, ymax2 = np.max(np.max(segments2, axis=0), axis=0)

        if xmax1 <= xmin2 or xmax2 <= xmin1 or ymax1 <= ymin2 or ymax2 <= ymin1:
            # segments can't intersect since their bounding rectangle don't intersect
            continue

        # check intersection over pairs of segments - O(n^2)

        # construct numpy array of shape (N = len(segments1) * len(segments2), 4, 2)
        # where each element contain 4 points ABCD (segment AB of segments1 and segment CD of segments2)
        idx1 = np.repeat(
            np.arange(len(segments1)),
            len(segments2))  # build indexes 1 1 1 2 2 2 3 3 3 4 4 4
        idx2 = np.tile(np.arange(len(segments2)),
                       len(segments1))  # build indexes 1 2 3 1 2 3 1 2 3 1 2 3
        segment_pairs = np.concatenate(
            (segments1[idx1], segments2[idx2]),
            axis=1)  # concatenate to create all pairs

        # now we need to check if at least one element ABCD contains an intersection between segment AB and segment CD
        def ccw(A, B, C):
            return (C[:, 1] - A[:, 1]) * (B[:, 0] - A[:, 0]) > (
                B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])

        # ABCD are each arrays of N points (shape (N, 2))
        A = segment_pairs[:, 0]
        B = segment_pairs[:, 1]
        C = segment_pairs[:, 2]
        D = segment_pairs[:, 3]
        if np.logical_and(
                ccw(A, C, D) != ccw(B, C, D),
                ccw(A, B, C) != ccw(A, B, D)).any():
            n_collisions[veh1] += 1
            n_collisions[veh2] += 1

    return n_collisions


def _intesection_metrics_impl(trajectory_path, model, configs):
    print(trajectory_path)

    scenario_config = configs['scenario_cfg']

    view_dist = configs['dataloader_cfg']['view_dist']
    view_angle = configs['dataloader_cfg']['view_angle']
    state_normalization = configs['dataloader_cfg']['state_normalization']
    dt = configs['dataloader_cfg']['dt']

    n_stacked_states = configs['dataloader_cfg']['n_stacked_states']
    state_size = configs['model_cfg']['n_inputs'] // n_stacked_states
    state_dict = defaultdict(lambda: np.zeros(state_size * n_stacked_states))

    # create model simulation
    sim = Simulation(str(trajectory_path), scenario_config)
    scenario = sim.getScenario()
    vehicles = scenario.getVehicles()
    objects = scenario.getObjectsThatMoved()

    # set all objects to be expert-controlled
    for obj in objects:
        obj.expert_control = True

    # in model sim, model will control vehicles that moved
    controlled_vehicles = [veh for veh in vehicles if veh in objects]

    # only control 2 vehicles at random
    random.shuffle(controlled_vehicles)
    # controlled_vehicles = controlled_vehicles[:2]

    # warmup to build up state stacking
    for i in range(n_stacked_states - 1):
        for veh in controlled_vehicles:
            ego_state = scenario.ego_state(veh)
            visible_state = scenario.flattened_visible_state(
                veh, view_dist=view_dist, view_angle=view_angle)
            state = np.concatenate(
                (ego_state, visible_state)) / state_normalization
            state_dict[veh.getID()] = np.roll(state_dict[veh.getID()],
                                              len(state))
            state_dict[veh.getID()][:len(state)] = state
        sim.step(dt)

    for veh in controlled_vehicles:
        veh.expert_control = False

    collisions = [False] * len(controlled_vehicles)
    goal_achieved = [False] * len(controlled_vehicles)
    for i in range(SIM_N_STEPS - n_stacked_states):
        for veh in controlled_vehicles:
            if np.isclose(veh.position.x, INVALID_POSITION):
                veh.expert_control = True
            else:
                veh.expert_control = False
        # set model actions
        # get all actions at once
        all_states = []
        for veh in controlled_vehicles:
            # get vehicle state
            state = np.concatenate(
                (scenario.ego_state(veh),
                 scenario.flattened_visible_state(
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

        # step simulation
        sim.step(dt)

        # compute displacements over non-collided vehicles
        for i, veh in enumerate(controlled_vehicles):
            # make sure it is valid
            if np.isclose(veh.position.x, INVALID_POSITION):
                continue

            # a collision with another a vehicle
            if veh.collided and int(veh.collision_type) == 1:
                collisions[i] = True
            if (veh.position - veh.target_position).norm() < GOAL_TOLERANCE:
                goal_achieved[i] = True

    # compute expert intersections for all vehicles (mapping veh_id -> nb of intersections in expert traj)
    intersection_data = _compute_expert_intersections(trajectory_path)

    # compute metrics as a function of number of intersections

    collision_rates = np.zeros(4)
    goal_rates = np.zeros(4)
    counts = np.zeros(4)
    for i, veh in enumerate(controlled_vehicles):
        n_intersections = min(intersection_data[veh.getID()], 3)
        counts[n_intersections] += 1
        if collisions[i]:
            collision_rates[n_intersections] += 1
        if goal_achieved[i]:
            goal_rates[n_intersections] += 1
    collision_rates /= counts
    goal_rates /= counts
    # note: returned values can contain NaN

    return collision_rates, goal_rates


def compute_metrics_by_intersection(trajectories_dir, model, configs):
    """Compute metrics as a function of number of intesections in a vehicle's expert trajectory."""
    NUM_FILES = 200
    NUM_CPUS = 14

    # get trajectories paths
    trajectories_dir = Path(trajectories_dir)
    trajectories_paths = list(trajectories_dir.glob('*tfrecord*.json'))
    trajectories_paths.sort()
    trajectories_paths = trajectories_paths[:NUM_FILES]

    # parallel metric computation
    with Pool(processes=NUM_CPUS) as pool:
        result = np.array(
            list(
                pool.starmap(
                    _intesection_metrics_impl,
                    zip(trajectories_paths, repeat(model), repeat(configs)))))
        assert result.shape == (len(trajectories_paths), 2, 4
                                )  # collision rates, goal rates (in 4 bins)
        avg_result = np.nanmean(result, axis=0)  # nanmean ignores NaN values
        print(avg_result)
        return avg_result


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
            '/checkpoint/eugenevinitsky/nocturne/test/2022.06.05/test/14.23.17\
                /++device=cuda,++file_limit=1000/train_logs/2022_06_05_14_23_23/configs.json',
            'r') as fp:
        configs = json.load(fp)
        configs['device'] = 'cpu'
    with torch.no_grad():
        result = compute_metrics_by_intersection(
            '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/\
                uncompressed/scenario/formatted_json_v2_no_tl_valid',
            model=model,
            configs=configs)
        print('collision rates', result[0])
        print('goal rates', result[1])
