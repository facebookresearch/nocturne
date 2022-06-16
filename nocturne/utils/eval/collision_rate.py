# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Collision rate computation."""
from pathlib import Path
import numpy as np
import torch

from nocturne import Simulation
from cfgs.config import ERR_VAL as INVALID_POSITION


SIM_N_STEPS = 90  # number of steps per trajectory
SIM_STEP_TIME = 0.1  # dt (in seconds)


def _collision_rate_impl(trajectory_path, model=None, sim_allow_non_vehicles=True, check_vehicles_only=True):
    # create expert simulation
    sim = Simulation(scenario_path=str(trajectory_path), start_time=0, allow_non_vehicles=sim_allow_non_vehicles)
    scenario = sim.getScenario()
    vehicles = scenario.getVehicles()
    objects_that_moved = scenario.getObjectsThatMoved()
    vehicles_that_moved = [veh for veh in vehicles if veh in objects_that_moved]

    # set all objects to be expert-controlled
    for obj in objects_that_moved:
        obj.expert_control = True
    for obj in vehicles:
        obj.expert_control = True

    # if a model is given, model will control vehicles that moved
    if model is not None:
        controlled_vehicles = vehicles_that_moved
        for veh in controlled_vehicles:
            veh.expert_control = False
    else:
        controlled_vehicles = []

    # vehicles to check for collisions on
    objects_to_check = [
        obj for obj in (vehicles_that_moved if check_vehicles_only else objects_that_moved)
        if (obj.target_position - obj.position).norm() > 0.5
    ]

    # step sim until the end and check for collisions
    collided_with_vehicle = {obj.id: False for obj in objects_to_check}
    collided_with_edge = {obj.id: False for obj in objects_to_check}
    for i in range(SIM_N_STEPS):
        # set model actions
        for veh in controlled_vehicles:
            # get vehicle state
            state = torch.as_tensor(np.expand_dims(np.concatenate(
                (scenario.ego_state(veh),
                 scenario.flattened_visible_state(veh, view_dist=120, view_angle=3.14))
            ), axis=0), dtype=torch.float32)
            # compute vehicle action
            action = model(state)[0]
            # set vehicle action
            veh.acceleration = action[0]
            veh.steering = action[1]

        # step simulation
        sim.step(SIM_STEP_TIME)

        # check for collisions
        for obj in objects_to_check:
            if not np.isclose(obj.position.x, INVALID_POSITION) and obj.collided:
                if int(obj.collision_type) == 1:
                    collided_with_vehicle[obj.id] = True
                if int(obj.collision_type) == 2:
                    collided_with_edge[obj.id] = True

    # compute collision rate
    collisions_with_vehicles = list(collided_with_vehicle.values())
    collisions_with_edges = list(collided_with_edge.values())
    collision_rate_vehicles = collisions_with_vehicles.count(True) / len(collisions_with_vehicles)
    collision_rate_edges = collisions_with_edges.count(True) / len(collisions_with_edges)

    return collision_rate_vehicles, collision_rate_edges


def compute_average_collision_rate(trajectories_dir, model=None, **kwargs):
    """Compute average collision rate for a model."""
    # get trajectories paths
    if isinstance(trajectories_dir, str):
        # if trajectories_dir is a string, treat it as the path to a directory of trajectories
        trajectories_dir = Path(trajectories_dir)
        trajectories_paths = list(trajectories_dir.glob('*tfrecord*.json'))
    elif isinstance(trajectories_dir, list):
        # if trajectories_dir is a list, treat it as a list of paths to trajectory files
        trajectories_paths = [Path(path) for path in trajectories_dir]
    # compute average collision rate over each individual trajectory file
    average_collision_rates = np.array(list(map(
        lambda path: _collision_rate_impl(path, model, **kwargs),
        trajectories_paths
    )))

    return np.mean(average_collision_rates, axis=0)


if __name__ == '__main__':
    from nocturne.utils.imitation_learning.waymo_data_loader import ImitationAgent  # noqa: F401
    model = torch.load('model.pth')
    collisions_with_vehicles, collisions_with_road_lines = \
        compute_average_collision_rate('dataset/json_files', model=None)
    print(f'Average Collision Rate: {100*collisions_with_vehicles:.2f}% with vehicles, '
          f'{100*collisions_with_road_lines:.2f}% with road lines')
