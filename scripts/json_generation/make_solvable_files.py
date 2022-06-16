# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Find all cases where collisions are required to achieve the goal.

Due to errors in Waymo labeling, some space that is crossable is mistakenly
labeled as a road edge. This file finds most of those cases.
"""
import argparse
import json
import multiprocessing
from multiprocessing import Process, Lock
import os

import hydra
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROCESSED_VALID_NO_TL, get_scenario_dict
from nocturne import Simulation


def is_file_valid(file_list, output_file, output_file_invalid, cfg, lock=None):
    """Test if file requires an agent to collide with a road edge to get to goal.

    We test for this by making the agent have very thin width. If an agent
    is in collision with a road edge despite this thin width, it was crossing
    that road edge because that road edge was on the way to its goal. We also
    shrink the length to avoid the cases where the vehicle is initialized
    in collision with a road edge.

    If a file has more than 80% of the agents need to collide with a road edge to get
    to goal, we store it in output_file_invalid instead.

    Args
    ----
        file_list ([str]): list of file paths.
        output_file (str): file to store valid json names.
        output_file_invalid (_type_): file to store invalid json names.
        lock (Lock, optional): Lock used for safe file writing.
    """
    file_valid_dict = {}
    file_invalid_dict = {}
    cfg['start_time'] = 0
    cfg['allow_non_vehicles'] = False
    for i, file in enumerate(file_list):
        sim = Simulation(str(file), get_scenario_dict(cfg))
        vehs = sim.scenario().getObjectsThatMoved()
        for veh in vehs:
            # we shrink the vehicle width and length to tiny values.
            # then, if a vehicle collides with a road edge, we know it had to
            # cross that road edge to actually get to its goal
            veh._scale_shape(length_scale=0.3, width_scale=0.1)
            veh.expert_control = True
        # dict tracking which vehicles were forced to collide with
        # an edge on their way to goal
        veh_edge_collided = {veh.id: False for veh in vehs}
        for _ in range(90):
            for veh in vehs:
                collided = veh.collided
                # the second conditions check whether the
                # the vehicle has "collided", but only because
                # it was invalid at the same time as another
                # vehicle was invalid
                if collided and not np.isclose(veh.position.x, -10000.0):
                    if int(veh.collision_type) == 2:
                        veh_edge_collided[veh.id] = True
            sim.step(0.1)
        # write all the vehicle ids that had a collision to a file
        # so that we know which vehicles should be set to be experts
        # if more than 80% of the vehicles are experts, we throw the file
        # away
        if np.sum(list(
                veh_edge_collided.values())) / len(veh_edge_collided) < 0.8:
            storage = file_valid_dict
        else:
            storage = file_invalid_dict
        storage[str(file).split('/')[-1]] = [
            key for key, val in veh_edge_collided.items() if val
        ]

    for file, return_dict in zip([output_file, output_file_invalid],
                                 [file_valid_dict, file_invalid_dict]):
        if lock is not None:
            lock.acquire()
        with open(file, 'r') as fp:
            temp_dict = json.load(fp)
        with open(file, 'w') as fp:
            temp_dict.update(return_dict)
            json.dump(temp_dict, fp, indent=4)
        if lock is not None:
            lock.release()


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """See file docstring."""
    disp = Display()
    disp.start()
    parser = argparse.ArgumentParser(
        description="Load and show waymo scenario data.")
    parser.add_argument(
        "--parallel",
        action='store_true',
        help="If true, split the conversion up over multiple processes")
    parser.add_argument(
        "--n_processes",
        type=int,
        default=40,
        help="Number of processes over which to split file generation")
    parser.add_argument("--datatype",
                        default='train',
                        type=str,
                        choices=['train', 'valid'],
                        nargs='+',
                        help="Whether to convert, train or valid data")

    args = parser.parse_args()
    # TODO(eugenevinitsky) this currently assumes that we have
    # constructed the scenes without traffic lights and not
    # other scenes
    folders_to_convert = []
    if 'train' in args.datatype:
        folders_to_convert.append(PROCESSED_TRAIN_NO_TL)
    if 'valid' in args.datatype:
        folders_to_convert.append(PROCESSED_VALID_NO_TL)

    lock = Lock()
    for folder_path in folders_to_convert:
        files = os.listdir(folder_path)
        files = [
            os.path.join(folder_path, file) for file in files
            if 'tfrecord' in file
        ]

        output_file = os.path.join(folder_path, 'valid_files.json')
        with open(output_file, 'w') as fp:
            json.dump({}, fp)

        output_file_invalid = os.path.join(folder_path, 'invalid_files.json')
        with open(output_file_invalid, 'w') as fp:
            json.dump({}, fp)

        if args.parallel:
            # leave some cpus free but have at least one and don't use more than n_processes
            num_cpus = min(max(multiprocessing.cpu_count() - 2, 1),
                           args.n_processes)
            num_files = len(files)
            process_list = []
            for i in range(num_cpus):
                p = Process(target=is_file_valid,
                            args=[
                                files[i * num_files // num_cpus:(i + 1) *
                                      num_files // num_cpus], output_file,
                                output_file_invalid, cfg, lock
                            ])
                p.start()
                process_list.append(p)

            for process in process_list:
                process.join()
        else:
            is_file_valid(files,
                          output_file,
                          output_file_invalid,
                          cfg,
                          lock=None)


if __name__ == '__main__':
    main()
