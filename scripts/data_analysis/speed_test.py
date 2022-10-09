# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utils that we use to understand the datasets we are working with."""
import json
import os
import time

import hydra
import numpy as np
import matplotlib.pyplot as plt

from cfgs.config import PROCESSED_TRAIN_NO_TL, get_scenario_dict, set_display_window
from nocturne import Simulation, Action


def run_speed_test(files, cfg):
    """Compute the expert accelerations and number of vehicles across the dataset.

    Args:
        files ([str]): List of files to analyze

    Returns
    -------
        [np.float], [np.float]: List of expert accels, list of number
                                of moving vehicles in file
    """
    times_list = np.zeros(200)
    count_list = np.zeros(200)
    speed_for_ten_vehs = []
    for file in files:
        for t in range(80):
            local_cfg = get_scenario_dict(cfg)
            local_cfg['start_time'] = t
            sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                            local_cfg)
            vehs = sim.scenario().getObjectsThatMoved()
            scenario = sim.getScenario()
            t = time.perf_counter()
            for veh in vehs:
                _ = scenario.flattened_visible_state(veh, 80, (120 / 180) * np.pi)
                veh.apply_action(Action(1.0, 1.0, 1.0))
            sim.step(0.1)
            if len(sim.scenario().getVehicles())  > 0 :
                times_list[len(sim.scenario().getVehicles()) - 1] += (time.perf_counter() - t)
                count_list[len(sim.scenario().getVehicles()) - 1] += len(vehs)
                if len(sim.scenario().getVehicles()) == 10:
                    speed_for_ten_vehs.append((time.perf_counter() - t) / len(vehs))

    print(1 / (times_list / count_list))
    print((1 / (times_list / count_list))[9])
    print(count_list[9])
    print('avg, std. time to get obs for scenes containing 10 vehicles is {}, {}'.format(np.mean(1 / np.array(speed_for_ten_vehs)),
                                                       np.std(1 / np.array(speed_for_ten_vehs))))
    plt.figure(dpi=300)
    plt.plot(np.linspace(1, 40, 40), (1 / (times_list / count_list))[0:40])
    plt.xlabel('Number of vehicles in scene')
    plt.ylabel('FPS')
    print('saving a figure in {}'.format(os.getcwd()))
    plt.savefig('fps_v_agent')


@hydra.main(config_path="../../cfgs/", config_name="config")
def analyze_accels(cfg):
    """Plot the expert accels and number of observed moving vehicles."""
    f_path = PROCESSED_TRAIN_NO_TL
    with open(os.path.join(f_path, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
    run_speed_test(files, cfg)


if __name__ == '__main__':
    set_display_window()
    analyze_accels()
