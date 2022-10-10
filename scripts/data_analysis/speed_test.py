# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utils that we use to understand the datasets we are working with."""
import json
import os
import pickle
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

    max_num_vehicles = 200
    num_time_steps = 90
    num_files = 10
    stats = {}

    # single agent computation
    times_list = np.zeros(max_num_vehicles)
    count_list = np.zeros(max_num_vehicles)

    cnt = 0
    for file in files:
        cnt += 1
        if cnt >= num_files:
            break

        for t in range(num_time_steps):
            local_cfg = get_scenario_dict(cfg)
            local_cfg['start_time'] = t
            sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                             local_cfg)
            scenario = sim.getScenario()

            vehs = scenario.vehicles()
            moving_vehs = scenario.moving_objects()
            num_vehs = len(vehs)
            num_moving_vehs = len(moving_vehs)
            if num_moving_vehs == 0:
                break

            num_objs = len(scenario.objects())
            if num_vehs != num_objs:
                print(f"[Single] num_vehs = {num_vehs}, num_objs = {num_objs}")

            agent_idx = np.random.randint(num_moving_vehs)
            for i, veh in enumerate(moving_vehs):
                if i != agent_idx:
                    veh.expert_control = True
            veh = moving_vehs[agent_idx]

            t = time.perf_counter_ns()
            _ = scenario.flattened_visible_state(veh, 80, (120 / 180) * np.pi)
            veh.apply_action(Action(1.0, 1.0, 1.0))
            sim.step(0.1)
            total_time = time.perf_counter_ns() - t

            times_list[num_vehs - 1] += total_time
            count_list[num_vehs - 1] += 1

    times_list *= 1e-9
    avg_sec = times_list / count_list
    avg_fps = 1.0 / avg_sec
    avg_sec = np.nan_to_num(avg_sec)
    avg_fps = np.nan_to_num(avg_fps)

    print(avg_sec)
    print(avg_fps)

    overall_avg_sec = np.sum(times_list) / np.sum(count_list)
    overall_avg_fps = 1.0 / overall_avg_sec
    print(f"overall_avg_sec = {overall_avg_sec}")
    print(f"overall_avg_fps = {overall_avg_fps}")

    stats["single_times_list"] = times_list
    stats["single_count_list"] = count_list
    stats["single_avg_sec"] = avg_sec
    stats["single_avg_fps"] = avg_fps

    # print(1 / (times_list / count_list))
    # print((1 / (times_list / count_list))[9])
    # print(count_list[9])
    # print(
    #     'avg, std. time to get obs for scenes containing 10 vehicles is {}, {}'
    #     .format(np.mean(1 / np.array(speed_for_ten_vehs)),
    #             np.std(1 / np.array(speed_for_ten_vehs))))
    # plt.figure(dpi=300)
    # plt.plot(np.linspace(1, 40, 40), (1 / (times_list / count_list))[0:40])
    # plt.xlabel('Number of vehicles in scene')
    # plt.ylabel('steps-per-second')
    # print('saving a figure in {}'.format(os.getcwd()))
    # plt.savefig('fps_v_agent')

    # multi-agent computation
    times_list = np.zeros(max_num_vehicles)
    count_list = np.zeros(max_num_vehicles)

    avg_agent_num = []
    cnt = 0
    for file in files:
        cnt += 1
        if cnt >= num_files:
            break

        for t in range(num_time_steps):
            local_cfg = get_scenario_dict(cfg)
            local_cfg['start_time'] = t
            sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                             local_cfg)
            scenario = sim.scenario()
            vehs = scenario.vehicles()
            moving_vehs = scenario.moving_objects()
            num_vehs = len(vehs)
            num_moving_vehs = len(moving_vehs)
            if num_moving_vehs == 0:
                break

            num_objs = len(scenario.objects())
            if num_vehs != num_objs:
                print(f"[Multi] num_vehs = {num_vehs}, num_objs = {num_objs}")

            avg_agent_num.append(num_moving_vehs)
            t = time.perf_counter_ns()
            for veh in moving_vehs:
                _ = scenario.flattened_visible_state(veh, 80,
                                                     (120 / 180) * np.pi)
                veh.apply_action(Action(1.0, 1.0, 1.0))
            sim.step(0.1)
            total_time = time.perf_counter_ns() - t

            times_list[num_vehs - 1] += total_time
            count_list[num_vehs - 1] += 1

    times_list *= 1e-9
    avg_sec = times_list / count_list
    avg_fps = 1.0 / avg_sec
    avg_sec = np.nan_to_num(avg_sec)
    avg_fps = np.nan_to_num(avg_fps)

    print(avg_sec)
    print(avg_fps)

    overall_avg_sec = np.sum(times_list) / np.sum(count_list)
    overall_avg_fps = 1.0 / overall_avg_sec
    print(f"overall_avg_sec = {overall_avg_sec}")
    print(f"overall_avg_fps = {overall_avg_fps}")

    stats["multi_times_list"] = times_list
    stats["multi_count_list"] = count_list
    stats["multi_avg_sec"] = avg_sec
    stats["multi_avg_fps"] = avg_fps
    stats["multi_avg_agent_num"] = np.mean(avg_agent_num)

    # print(1 / (times_list / count_list))
    # print((1 / (times_list / count_list))[9])
    # plt.figure(dpi=300)
    # plt.plot(np.linspace(1, 40, 40), (1 / (times_list / count_list))[0:40])
    # plt.xlabel('Number of vehicles in scene')
    # plt.ylabel('steps-per-second')
    # print('saving a figure in {}'.format(os.getcwd()))
    # print('the average number of controlled agents is {}'.format(
    #     np.mean(avg_agent_num)))
    # plt.savefig('fps_v_agent_ma')

    with open("./perf_stats.pkl", 'wb') as f:
        pickle.dump(stats, f)


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
