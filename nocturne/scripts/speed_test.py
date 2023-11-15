# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utils that we use to understand the datasets we are working with."""
import json
import os
import time
from tqdm import tqdm
import hydra
import numpy as np
from pyvirtualdisplay import Display
from omegaconf import OmegaConf

from nocturne import Simulation, Action

DATA_FOLDER = '/home/aarav/nocturne_data/formatted_json_v2_no_tl_train'
VERSION_NUMBER = 2

PROCESSED_TRAIN_NO_TL = '/home/aarav/nocturne_data/formatted_json_v2_no_tl_train'

def get_scenario_dict(hydra_cfg):
    """Convert the `scenario` key in the hydra config to a true dict."""
    if isinstance(hydra_cfg['scenario'], dict):
        return hydra_cfg['scenario']
    else:
        return OmegaConf.to_container(hydra_cfg['scenario'], resolve=True)

def set_display_window():
    """Set a virtual display for headless machines."""
    if "DISPLAY" not in os.environ:
        disp = Display()
        disp.start()

def run_speed_test(files, cfg):
    """Compute the expert accelerations and number of vehicles across the dataset.

    Args:
        files ([str]): List of files to analyze

    Returns
    -------
        [np.float], [np.float]: List of expert accels, list of number
                                of moving vehicles in file
    """
    times_list = []
    for file in tqdm(files):
        sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                         get_scenario_dict(cfg))
        vehs = sim.scenario().getObjectsThatMoved()
        scenario = sim.getScenario()
        veh = vehs[np.random.randint(len(vehs))]
        t = time.perf_counter()
        _ = scenario.flattened_visible_state(veh, 80, (180 / 180) * np.pi)
        veh.apply_action(Action(1.0, 1.0, 1.0))
        sim.step(0.1)
        times_list.append(time.perf_counter() - t)
    print('avg, std. time to get obs is {}, {}'.format(np.mean(times_list),
                                                       np.std(times_list)))


@hydra.main(config_path="../../configs", config_name="env_config")
def analyze_accels(cfg):
    """Plot the expert accels and number of observed moving vehicles."""
    f_path = PROCESSED_TRAIN_NO_TL
    with open(os.path.join(f_path, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
    run_speed_test(files[:1000], cfg)


if __name__ == '__main__':
    set_display_window()
    analyze_accels()