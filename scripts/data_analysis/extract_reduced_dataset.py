# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run through the data and extract trajectories of all the agents
"""
from copy import deepcopy
from collections import defaultdict
from pathlib import Path
import pickle as pkl
import os
import sys

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH, \
    get_scenario_dict, set_display_window, ERR_VAL
from nocturne import Simulation


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """See file docstring."""
    MAX_NUM_VEHICLES = 2
    set_display_window()
    files = list(os.listdir(PROCESSED_TRAIN_NO_TL))
    files = [file for file in files if 'tfrecord' in file]
    # data storage
    data_dict = defaultdict(dict)

    start_cfg = deepcopy(cfg)
    start_cfg['scenario']['start_time'] = 0
    start_cfg['scenario']['allow_non_vehicles'] = False
    veh_counter = 0
    for file_idx, file in enumerate(files):
        sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                         get_scenario_dict(cfg))
        vehs = sim.getScenario().getVehicles()
        for veh in vehs:
            veh.expert_control = True
        if len(sim.getScenario().moving_objects()) < 3:
            continue
        data_dict[file]['trajectory'] = {veh.getID(): -100 * np.ones((2 + 4 * (MAX_NUM_VEHICLES + 1), 90)) for veh in sim.getScenario().moving_objects()}
        if file_idx < 100:
            img = sim.getScenario().getImage(
                img_width=1600,
                img_height=1600,
                draw_target_positions=True,
                padding=50.0,
            )
            data_dict[file]['scene_img'] = img
        veh_counter += len(sim.getScenario().moving_objects())
        for time_index in range(90):
            for veh in sim.getScenario().moving_objects():
                if np.isclose(veh.position.x, ERR_VAL):
                    continue
                obs_list = sim.getScenario().visible_vehicles(veh, 80, 3.14)
                # store the state of the vehicle
                data_dict[file]['trajectory'][veh.getID()][0:6, time_index] = [veh.target_position.x, veh.target_position.y,
                                                     veh.position.x, veh.position.y, veh.heading, veh.speed]

                for i, curr_veh in enumerate(obs_list[0:MAX_NUM_VEHICLES]):
                    try:
                        data_dict[file]['trajectory'][veh.getID()][6 + i * 4: 6 + (i + 1) * 4, time_index] = \
                        [curr_veh.position.x, curr_veh.position.y, curr_veh.heading, curr_veh.speed]
                    except:
                        import ipdb; ipdb.set_trace()
            sim.step(0.1)

    with open('expert_traj.pkl', 'wb') as file:
        pkl.dump(data_dict, file)
    print('output the results file in ', os.getcwd())
    print('there were {} files'.format(len(data_dict)))
    print('the total number of agent trajectories is {}'.format(veh_counter))

if __name__ == '__main__':
    main()
