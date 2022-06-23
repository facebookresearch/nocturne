# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run through the data to look for cases where there are undesirable corner cases.

The cases we currently check for are:
1) is a vehicle initialized in a colliding state with another vehicle
2) is a vehicle initialized in a colliding state with a road edge?
"""
from copy import deepcopy
from pathlib import Path
import os
import sys

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH, get_scenario_dict, set_display_window
from nocturne import Simulation


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """See file docstring."""
    set_display_window()
    SAVE_IMAGES = False
    MAKE_MOVIES = False
    output_folder = 'corner_case_vis'
    output_path = Path(PROJECT_PATH) / f'nocturne_utils/{output_folder}'
    output_path.mkdir(exist_ok=True)
    files = list(os.listdir(PROCESSED_TRAIN_NO_TL))
    files = [file for file in files if 'tfrecord' in file]
    # track the number of collisions at each time-step
    collide_counter = np.zeros((2, 90))
    file_has_veh_collision_counter = 0
    file_has_edge_collision_counter = 0
    total_edge_collision_counter = 0
    total_veh_collision_counter = 0
    initialized_collision_counter = 0
    total_veh_counter = 0

    start_cfg = deepcopy(cfg)
    start_cfg['scenario']['start_time'] = 0
    start_cfg['scenario']['allow_non_vehicles'] = False
    for file_idx, file in enumerate(files):
        found_collision = False
        edge_collision = False
        sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                         get_scenario_dict(cfg))
        vehs = sim.getScenario().getObjectsThatMoved()
        # this checks if the vehicles has actually moved any distance at all
        valid_vehs = []
        for veh in vehs:
            veh.expert_control = True
            obj_pos = veh.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            if np.linalg.norm(obj_pos - goal_pos) > 0.5:
                valid_vehs.append(veh)
        veh_edge_collided = [False for _ in vehs]
        veh_veh_collided = [False for _ in vehs]
        initialized_collided = [False for _ in vehs]
        for time_index in range(90):
            for veh_index, veh in enumerate(valid_vehs):
                collided = veh.getCollided()
                if collided and not np.isclose(veh.getPosition().x, -10000.0):
                    collide_counter[int(veh.collision_type) - 1,
                                    time_index] += 1
                    if int(veh.collision_type) == 2:
                        veh_edge_collided[veh_index] = True
                    if int(veh.collision_type) == 1:
                        veh_veh_collided[veh_index] = True
                    if time_index == 0:
                        initialized_collided[veh_index] = True
                if np.isclose(veh.getPosition().x, -10000.0):
                    collided = False
                if time_index == 0 and not found_collision and collided and SAVE_IMAGES:
                    img = sim.getScenario().getImage(
                        img_width=1600,
                        img_height=1600,
                        draw_target_positions=True,
                        padding=50.0,
                    )
                    fig = plt.figure()
                    plt.imshow(img)
                    plt.savefig(f'{output_folder}/{file}.png')
                    plt.close(fig)
                if not found_collision and collided:
                    found_collision = True
                    if int(veh.collision_type) == 1:
                        file_has_veh_collision_counter += 1
                    else:
                        file_has_edge_collision_counter += 1
                        edge_collision = True
            sim.step(0.1)
        total_veh_counter += len(valid_vehs)
        total_edge_collision_counter += np.sum(veh_edge_collided)
        total_veh_collision_counter += np.sum(veh_veh_collided)
        initialized_collision_counter += np.sum(initialized_collided)
        print(f'at file {file_idx} we have {collide_counter} collisions for a\
                 ratio of {collide_counter / (file_idx + 1)}')
        print(f'the number of files that have a veh collision at all is\
                 {file_has_veh_collision_counter / (file_idx + 1)}')
        print(f'the number of files that have a edge collision at all is\
                 {file_has_edge_collision_counter / (file_idx + 1)}')
        print(f'the fraction of vehicles that have had an edge collision\
                is {total_edge_collision_counter / total_veh_counter}')
        print(f'the fraction of vehicles that have had a collision at all\
                is {(total_edge_collision_counter + total_veh_collision_counter) / total_veh_counter}'
              )
        print(
            f'the fraction of vehicles that are initialized in collision are \
                {initialized_collision_counter / total_veh_counter}')
        if found_collision and edge_collision and MAKE_MOVIES:
            movie_frames = []
            fig = plt.figure()
            sim = Simulation(os.path.join(PROCESSED_TRAIN_NO_TL, file),
                             get_scenario_dict(start_cfg))
            vehs = sim.getScenario().getObjectsThatMoved()
            for veh in vehs:
                veh.expert_control = True
            for time_index in range(89):
                movie_frames.append(sim.getScenario().getImage(
                    img_width=1600, img_height=1600))
                sim.step(0.1)
            movie_frames = np.array(movie_frames)
            imageio.mimwrite(f'{output_path}/{os.path.basename(file)}.mp4',
                             movie_frames,
                             fps=10)
            if file_has_edge_collision_counter + file_has_veh_collision_counter > 10:
                sys.exit()


if __name__ == '__main__':
    main()
