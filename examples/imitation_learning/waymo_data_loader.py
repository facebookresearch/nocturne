# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Dataloader for imitation learning in Nocturne."""
from collections import defaultdict
import random

import torch
from pathlib import Path
import numpy as np

from cfgs.config import ERR_VAL
from nocturne import Simulation


def _get_waymo_iterator(paths, dataloader_config, scenario_config):
    # if worker has no paths, return an empty iterator
    if len(paths) == 0:
        return

    # load dataloader config
    tmin = dataloader_config.get('tmin', 0)
    tmax = dataloader_config.get('tmax', 90)
    view_dist = dataloader_config.get('view_dist', 80)
    view_angle = dataloader_config.get('view_angle', np.radians(120))
    dt = dataloader_config.get('dt', 0.1)
    expert_action_bounds = dataloader_config.get('expert_action_bounds',
                                                 [[-3, 3], [-0.7, 0.7]])
    expert_position = dataloader_config.get('expert_position', True)
    state_normalization = dataloader_config.get('state_normalization', 100)
    n_stacked_states = dataloader_config.get('n_stacked_states', 5)

    while True:
        # select a random scenario path
        scenario_path = np.random.choice(paths)

        # create simulation
        sim = Simulation(str(scenario_path), scenario_config)
        scenario = sim.getScenario()

        # set objects to be expert-controlled
        for obj in scenario.getObjects():
            obj.expert_control = True

        # we are interested in imitating vehicles that moved
        objects_that_moved = scenario.getObjectsThatMoved()
        objects_of_interest = [
            obj for obj in scenario.getVehicles() if obj in objects_that_moved
        ]

        # initialize values if stacking states
        stacked_state = defaultdict(lambda: None)
        initial_warmup = n_stacked_states - 1

        state_list = []
        action_list = []

        # iterate over timesteps and objects of interest
        for time in range(tmin, tmax):
            for obj in objects_of_interest:
                # get state
                ego_state = scenario.ego_state(obj)
                visible_state = scenario.flattened_visible_state(
                    obj, view_dist=view_dist, view_angle=view_angle)
                state = np.concatenate((ego_state, visible_state))

                # normalize state
                state /= state_normalization

                # stack state
                if n_stacked_states > 1:
                    if stacked_state[obj.getID()] is None:
                        stacked_state[obj.getID()] = np.zeros(
                            len(state) * n_stacked_states, dtype=state.dtype)
                    stacked_state[obj.getID()] = np.roll(
                        stacked_state[obj.getID()], len(state))
                    stacked_state[obj.getID()][:len(state)] = state

                if np.isclose(obj.position.x, ERR_VAL):
                    continue

                if not expert_position:
                    # get expert action
                    expert_action = scenario.expert_action(obj, time)
                    # check for invalid action (because no value available for taking derivative)
                    # or because the vehicle is at an invalid state
                    if expert_action is None:
                        continue
                    expert_action = expert_action.numpy()
                    # now find the corresponding expert actions in the grids

                    # throw out actions containing NaN or out-of-bound values
                    if np.isnan(expert_action).any() \
                            or expert_action[0] < expert_action_bounds[0][0] \
                            or expert_action[0] > expert_action_bounds[0][1] \
                            or expert_action[1] < expert_action_bounds[1][0] \
                            or expert_action[1] > expert_action_bounds[1][1]:
                        continue
                else:
                    expert_pos_shift = scenario.expert_pos_shift(obj, time)
                    if expert_pos_shift is None:
                        continue
                    expert_pos_shift = expert_pos_shift.numpy()
                    expert_heading_shift = scenario.expert_heading_shift(
                        obj, time)
                    if expert_heading_shift is None \
                            or expert_pos_shift[0] < expert_action_bounds[0][0] \
                            or expert_pos_shift[0] > expert_action_bounds[0][1] \
                            or expert_pos_shift[1] < expert_action_bounds[1][0] \
                            or expert_pos_shift[1] > expert_action_bounds[1][1] \
                            or expert_heading_shift < expert_action_bounds[2][0] \
                            or expert_heading_shift > expert_action_bounds[2][1]:
                        continue
                    expert_action = np.concatenate(
                        (expert_pos_shift, [expert_heading_shift]))

                # yield state and expert action
                if stacked_state[obj.getID()] is not None:
                    if initial_warmup <= 0:  # warmup to wait for stacked state to be filled up
                        state_list.append(stacked_state[obj.getID()])
                        action_list.append(expert_action)
                else:
                    state_list.append(state)
                    action_list.append(expert_action)

            # step the simulation
            sim.step(dt)
            if initial_warmup > 0:
                initial_warmup -= 1

        if len(state_list) > 0:
            temp = list(zip(state_list, action_list))
            random.shuffle(temp)
            state_list, action_list = zip(*temp)
            for state_return, action_return in zip(state_list, action_list):
                yield (state_return, action_return)


class WaymoDataset(torch.utils.data.IterableDataset):
    """Waymo dataset loader."""

    def __init__(self,
                 data_path,
                 dataloader_config={},
                 scenario_config={},
                 file_limit=None):
        super(WaymoDataset).__init__()

        # save configs
        self.dataloader_config = dataloader_config
        self.scenario_config = scenario_config

        # get paths of dataset files (up to file_limit paths)
        self.file_paths = list(
            Path(data_path).glob('tfrecord*.json'))[:file_limit]
        print(f'WaymoDataset: loading {len(self.file_paths)} files.')

        # sort the paths for reproducibility if testing on a small set of files
        self.file_paths.sort()

    def __iter__(self):
        """Partition files for each worker and return an (state, expert_action) iterable."""
        # get info on current worker process
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # single-process data loading, return the whole set of files
            return _get_waymo_iterator(self.file_paths, self.dataloader_config,
                                       self.scenario_config)

        # distribute a unique set of file paths to each worker process
        worker_file_paths = np.array_split(
            self.file_paths, worker_info.num_workers)[worker_info.id]
        return _get_waymo_iterator(list(worker_file_paths),
                                   self.dataloader_config,
                                   self.scenario_config)


if __name__ == '__main__':
    dataset = WaymoDataset(data_path='dataset/tf_records',
                           file_limit=20,
                           dataloader_config={
                               'view_dist': 80,
                               'n_stacked_states': 3,
                           },
                           scenario_config={
                               'start_time': 0,
                               'allow_non_vehicles': True,
                               'spawn_invalid_objects': True,
                           })

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        num_workers=4,
        pin_memory=True,
    )

    for i, x in zip(range(100), data_loader):
        print(i, x[0].shape, x[1].shape)
