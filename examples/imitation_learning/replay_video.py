# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Replay a video of a trained controller."""
from collections import defaultdict
import json
from pathlib import Path
import sys

import imageio
import numpy as np
from pyvirtualdisplay import Display
import subprocess
import torch

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH
from nocturne import Simulation, Vector2D

OUTPUT_PATH = str(PROJECT_PATH / 'vids')

MODEL_PATH = Path(
    '/checkpoint/eugenevinitsky/nocturne/test/2022.06.05/test/14.23.17/\
        ++device=cuda,++file_limit=1000/train_logs/2022_06_05_14_23_23/model_220.pth'
)
CONFIG_PATH = MODEL_PATH.parent / 'configs.json'
GOAL_TOLERANCE = 1.0

if __name__ == '__main__':
    disp = Display()
    disp.start()
    output_dir = Path(OUTPUT_PATH)
    output_dir.mkdir(exist_ok=True)

    with open(CONFIG_PATH, 'r') as f:
        configs = json.load(f)

    data_path = PROCESSED_TRAIN_NO_TL
    files = [
        file for file in Path(data_path).iterdir() if 'tfrecord' in file.stem
    ]
    scenario_config = configs['scenario_cfg']
    dataloader_config = configs['dataloader_cfg']
    files = files[:600]
    np.random.shuffle(files)
    model = torch.load(MODEL_PATH).to('cpu')
    model.eval()
    for traj_path in files:
        sim = Simulation(str(traj_path), scenario_config)
        output_str = traj_path.stem.split('.')[0].split('/')[-1]

        def policy(state):
            """Get model output."""
            state = torch.as_tensor(np.array([state]), dtype=torch.float32)
            return model.forward(state,
                                 deterministic=True,
                                 return_indexes=False)

        with torch.no_grad():
            for expert_control_vehicles, mp4_name in [
                (False, f'{output_str}_policy_rollout.mp4'),
                (True, f'{output_str}_true_rollout.mp4')
            ]:
                frames = []
                sim.reset()
                scenario = sim.getScenario()

                objects_of_interest = [
                    obj for obj in scenario.getVehicles()
                    if obj in scenario.getObjectsThatMoved()
                ]

                for obj in objects_of_interest:
                    obj.expert_control = True

                relevant_obj_ids = [
                    obj.getID() for obj in objects_of_interest[0:2]
                ]

                view_dist = configs['dataloader_cfg']['view_dist']
                view_angle = configs['dataloader_cfg']['view_angle']
                state_normalization = configs['dataloader_cfg'][
                    'state_normalization']
                dt = configs['dataloader_cfg']['dt']

                n_stacked_states = configs['dataloader_cfg'][
                    'n_stacked_states']
                state_size = configs['model_cfg'][
                    'n_inputs'] // n_stacked_states
                state_dict = defaultdict(
                    lambda: np.zeros(state_size * n_stacked_states))
                for i in range(n_stacked_states):
                    for veh in objects_of_interest:
                        ego_state = scenario.ego_state(veh)
                        visible_state = scenario.flattened_visible_state(
                            veh, view_dist=view_dist, view_angle=view_angle)
                        state = np.concatenate(
                            (ego_state, visible_state)) / state_normalization
                        state_dict[veh.getID()] = np.roll(
                            state_dict[veh.getID()], len(state))
                        state_dict[veh.getID()][:len(state)] = state

                    sim.step(dt)

                for obj in scenario.getObjectsThatMoved():
                    obj.expert_control = True
                # we only actually want to take control once the vehicle
                # has been placed into the network
                for veh in objects_of_interest:
                    if np.isclose(veh.position.x, -10000.0):
                        veh.expert_control = True
                    else:
                        if veh.getID() in relevant_obj_ids:
                            veh.expert_control = expert_control_vehicles
                            veh.highlight = True

                for i in range(90 - n_stacked_states):
                    # we only actually want to take control once the vehicle
                    # has been placed into the network
                    # so vehicles that should be controlled by our agent
                    # are overriden to be expert controlled
                    # until they are actually spawned in the scene
                    for veh in objects_of_interest:
                        if np.isclose(veh.position.x, -10000.0):
                            veh.expert_control = True
                        else:
                            if veh.getID() in relevant_obj_ids:
                                veh.expert_control = expert_control_vehicles
                                veh.highlight = True
                    print(
                        f'...{i+1}/{90 - n_stacked_states} ({traj_path} ; {mp4_name})'
                    )
                    img = scenario.getImage(
                        img_width=1600,
                        img_height=1600,
                        draw_target_positions=True,
                        padding=50.0,
                    )
                    frames.append(img)
                    for veh in objects_of_interest:
                        veh_state = np.concatenate(
                            (np.array(scenario.ego_state(veh), copy=False),
                             np.array(scenario.flattened_visible_state(
                                 veh,
                                 view_dist=view_dist,
                                 view_angle=view_angle),
                                      copy=False)))
                        ego_state = scenario.ego_state(veh)
                        visible_state = scenario.flattened_visible_state(
                            veh, view_dist=view_dist, view_angle=view_angle)
                        state = np.concatenate(
                            (ego_state, visible_state)) / state_normalization
                        state_dict[veh.getID()] = np.roll(
                            state_dict[veh.getID()], len(state))
                        state_dict[veh.getID()][:len(state)] = state
                        action = policy(state_dict[veh.getID()])
                        if dataloader_config['expert_position']:
                            if configs['model_cfg']['discrete']:
                                pos_diff = np.array([
                                    pos.cpu().numpy()[0] for pos in action[0:2]
                                ])
                                heading = action[2:3][0].cpu().numpy()[0]
                            else:
                                pos_diff = action[0:2]
                                heading = action[2:3]
                            veh.position = Vector2D.from_numpy(
                                pos_diff + veh.position.numpy())
                            veh.heading += heading
                        else:
                            veh.acceleration = action[0].cpu().numpy()
                            veh.steering = action[1].cpu().numpy()
                    sim.step(dt)
                    for veh in scenario.getObjectsThatMoved():
                        if (veh.position -
                                veh.target_position).norm() < GOAL_TOLERANCE:
                            scenario.removeVehicle(veh)
                imageio.mimsave(mp4_name, np.stack(frames, axis=0), fps=30)
                print(f'> {mp4_name}')

        # stack the movies side by side
        output_name = traj_path.stem.split('.')[0].split('/')[-1]
        output_path = f'{output_name}_output.mp4'
        ffmpeg_command = f'ffmpeg -y -i {output_str}_true_rollout.mp4 ' \
            f'-i {output_str}_policy_rollout.mp4 -filter_complex hstack {output_path}'
        print(ffmpeg_command)
        subprocess.call(ffmpeg_command.split(' '))
        print(f'> {output_path}')
        sys.exit()
