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

from typing import Any, Dict, Sequence, Union

from cfgs.config import PROCESSED_TRAIN_NO_TL, get_scenario_dict, set_display_window
from nocturne import Simulation, Action

MAX_NUM_VEHICLES = 400


def single_agent_test(cfg: Dict[str,
                                Any], files: Sequence[str], num_files: int,
                      num_steps: int) -> Dict[str, Union[float, np.ndarray]]:
    sec_by_veh = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    fps_by_veh = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    cnt_by_veh = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    avg_agt_num = []
    avg_veh_num = []

    cnt = 0
    for file in files:
        cnt += 1
        if cnt >= num_files:
            break

        for t in range(num_steps):
            local_cfg = get_scenario_dict(cfg)
            local_cfg['start_time'] = t
            sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                             local_cfg)
            scenario = sim.getScenario()

            vehs = scenario.vehicles()
            agts = scenario.moving_objects()
            num_vehs = len(vehs)
            num_agts = len(agts)
            if num_agts == 0:
                break

            num_objs = len(scenario.objects())
            if num_vehs != num_objs:
                print(f"[Single] num_vehs = {num_vehs}, num_objs = {num_objs}")

            avg_agt_num.append(num_agts)
            avg_veh_num.append(num_vehs)

            agt_idx = np.random.randint(num_agts)
            for i, agt in enumerate(agts):
                if i != agt_idx:
                    agt.expert_control = True
            agt = agts[agt_idx]

            try:
                t = time.perf_counter_ns()
                _ = scenario.flattened_visible_state(agt, 80,
                                                     (120 / 180) * np.pi)
                agt.apply_action(Action(1.0, 1.0, 1.0))
                sim.step(0.1)
                total_time = time.perf_counter_ns() - t
            except Exception as e:
                print(e)
                continue

            if num_vehs <= MAX_NUM_VEHICLES:
                sec_by_veh[num_vehs - 1] += total_time
                fps_by_veh[num_vehs - 1] += 1 / (total_time * 1e-9)
                cnt_by_veh[num_vehs - 1] += 1

    sec_by_veh = sec_by_veh * 1e-9
    avg_sec = sec_by_veh / cnt_by_veh
    avg_fps = fps_by_veh / cnt_by_veh
    avg_sec = np.nan_to_num(avg_sec)
    avg_fps = np.nan_to_num(avg_fps)

    overall_avg_sec = np.sum(sec_by_veh) / np.sum(cnt_by_veh)
    overall_avg_fps = np.sum(fps_by_veh) / np.sum(cnt_by_veh)
    avg_agt_num = np.mean(avg_agt_num)
    avg_veh_num = np.mean(avg_veh_num)

    print(f"[single] avg_sec = {avg_sec}")
    print(f"[single] avg_fps = {avg_fps}")
    print(f"[single] overall_avg_sec = {overall_avg_sec}")
    print(f"[single] overall_avg_fps = {overall_avg_fps}")
    print(f"[single] overall_avg_veh = {avg_veh_num}")
    print(f"[single] overall_avg_agt = {avg_agt_num}")

    return {
        "single_sec_by_veh": sec_by_veh,
        "single_cnt_by_veh": cnt_by_veh,
        "single_avg_sec": avg_sec,
        "single_avg_fps": avg_fps,
        "single_avg_agt_num": avg_agt_num,
        "single_avg_veh_num": avg_veh_num,
    }


def multi_agent_test(cfg: Dict[str, Any], files: Sequence[str], num_files: int,
                     num_steps: int) -> Dict[str, Union[float, np.ndarray]]:
    sec_by_veh = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    fps_by_veh = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    cnt_by_veh = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    sec_by_agt = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    fps_by_agt = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    cnt_by_agt = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)
    veh_by_agt = np.zeros(MAX_NUM_VEHICLES, dtype=np.int64)

    avg_agt_num = []
    avg_veh_num = []

    cnt = 0
    for file in files:
        cnt += 1
        if cnt >= num_files:
            break

        for t in range(num_steps):
            local_cfg = get_scenario_dict(cfg)
            local_cfg['start_time'] = t
            sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                             local_cfg)
            scenario = sim.scenario()
            vehs = scenario.vehicles()
            agts = scenario.moving_objects()
            num_vehs = len(vehs)
            num_agts = len(agts)
            if num_agts == 0:
                break

            num_objs = len(scenario.objects())
            if num_vehs != num_objs:
                print(f"[Multi] num_vehs = {num_vehs}, num_objs = {num_objs}")

            avg_agt_num.append(num_agts)
            avg_veh_num.append(num_vehs)

            try:
                t = time.perf_counter_ns()
                for agt in agts:
                    _ = scenario.flattened_visible_state(
                        agt, 80, (120 / 180) * np.pi)
                    agt.apply_action(Action(1.0, 1.0, 1.0))
                sim.step(0.1)
                total_time = time.perf_counter_ns() - t
            except Exception as e:
                print(e)
                continue

            if num_vehs <= MAX_NUM_VEHICLES:
                sec_by_veh[num_vehs - 1] += total_time
                fps_by_veh[num_vehs - 1] += 1 / (total_time * 1e-9)
                cnt_by_veh[num_vehs - 1] += 1
                sec_by_agt[num_agts - 1] += total_time
                fps_by_agt[num_agts - 1] += 1 / (total_time * 1e-9)
                cnt_by_agt[num_agts - 1] += 1
                veh_by_agt[num_agts - 1] += num_vehs

    sec_by_veh = sec_by_veh * 1e-9
    sec_by_agt = sec_by_agt * 1e-9
    avg_sec_by_veh = sec_by_veh / cnt_by_veh
    avg_sec_by_agt = sec_by_agt / cnt_by_agt
    avg_fps_by_veh = fps_by_veh / cnt_by_veh
    avg_fps_by_agt = fps_by_agt / cnt_by_agt
    avg_veh_by_agt = veh_by_agt / cnt_by_agt

    avg_sec_by_veh = np.nan_to_num(avg_sec_by_veh)
    avg_fps_by_veh = np.nan_to_num(avg_fps_by_veh)
    avg_sec_by_agt = np.nan_to_num(avg_sec_by_agt)
    avg_fps_by_agt = np.nan_to_num(avg_fps_by_agt)
    avg_veh_by_agt = np.nan_to_num(avg_veh_by_agt)

    overall_avg_sec = np.sum(sec_by_veh) / np.sum(cnt_by_veh)
    overall_avg_fps = 1.0 / overall_avg_sec
    avg_agt_num = np.mean(avg_agt_num)
    avg_veh_num = np.mean(avg_veh_num)

    print(f"[multi] avg_sec_by_veh = {avg_sec_by_veh}")
    print(f"[multi] avg_fps_by_veh = {avg_fps_by_veh}")
    print(f"[multi] avg_sec_by_agt = {avg_sec_by_agt}")
    print(f"[multi] avg_fps_by_agt = {avg_fps_by_agt}")
    print(f"[multi] avg_veh_by_agt = {avg_veh_by_agt}")
    print(f"[multi] overall_avg_sec = {overall_avg_sec}")
    print(f"[multi] overall_avg_fps = {overall_avg_fps}")
    print(f"[multi] overall_avg_veh = {avg_veh_num}")
    print(f"[multi] overall_avg_agt = {avg_agt_num}")

    return {
        "multi_sec_by_veh": sec_by_veh,
        "multi_cnt_by_veh": cnt_by_veh,
        "multi_sec_by_agt": sec_by_agt,
        "multi_cnt_by_agt": cnt_by_agt,
        "multi_veh_by_agt": veh_by_agt,
        "multi_avg_sec_by_veh": avg_sec_by_veh,
        "multi_avg_fps_by_veh": avg_fps_by_veh,
        "multi_avg_sec_by_agt": avg_sec_by_agt,
        "multi_avg_fps_by_agt": avg_fps_by_agt,
        "multi_avg_veh_by_agt": avg_veh_by_agt,
        "multi_avg_agt_num": avg_agt_num,
        "multi_avg_veh_num": avg_veh_num,
    }


def run_speed_test(files, cfg):
    """Compute the expert accelerations and number of vehicles across the dataset.

    Args:
        files ([str]): List of files to analyze

    Returns
    -------
        [np.float], [np.float]: List of expert accels, list of number
                                of moving vehicles in file
    """

    num_files = 1000
    num_steps = 90
    stats = {}

    stats1 = single_agent_test(cfg, files, num_files, num_steps)
    stats.update(stats1)

    stats2 = multi_agent_test(cfg, files, num_files, num_steps)
    stats.update(stats2)

    print(stats)

    with open("./perf_stats.pkl", 'wb') as f:
        pickle.dump(stats, f)


@hydra.main(config_path="../../cfgs/", config_name="config")
def analyze_accels(cfg):
    """Plot the expert accels and number of observed moving vehicles."""
    f_path = PROCESSED_TRAIN_NO_TL
    with open(os.path.join(f_path, 'valid_files.json')) as file:
        valid_veh_dict = json.load(file)
        files = list(valid_veh_dict.keys())
    print(f"tot_files = {len(files)}")
    run_speed_test(files, cfg)


if __name__ == '__main__':
    set_display_window()
    analyze_accels()
