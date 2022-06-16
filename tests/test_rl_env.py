# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test step and rendering functions."""
import time
import os

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROJECT_PATH
from nocturne import Action
from nocturne.envs.wrappers import create_env


def test_rl_env():
    """Test step and rendering functions."""
    disp = Display()
    disp.start()
    GlobalHydra.instance().clear()
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    cfg.scenario_path = os.path.join(PROJECT_PATH, 'tests')
    cfg.max_num_vehicles = 50
    env = create_env(cfg)
    env.files = [str(PROJECT_PATH / "tests/large_file.json")]
    times = []
    _ = env.reset()
    # quick check that rendering works
    _ = env.scenario.getConeImage(env.scenario.getVehicles()[0], 120.0,
                                  1.99 * 3.14, 0.0, draw_target_position=False)
    for _ in range(90):
        vehs = env.scenario.getObjectsThatMoved()
        prev_position = {
            veh.getID(): [veh.position.x, veh.position.y]
            for veh in vehs
        }
        t = time.perf_counter()
        obs, rew, done, info = env.step(
            {veh.id: Action(acceleration=2.0, steering=1.0)
             for veh in vehs})
        times.append(time.perf_counter() - t)
        for veh in vehs:
            if veh in env.scenario.getObjectsThatMoved():
                new_position = [veh.position.x, veh.position.y]
                assert prev_position[veh.getID(
                )] != new_position, f'veh {veh.getID()} was in position \
                    {prev_position[veh.getID()]} which is the \
                        same as {new_position} but should have moved'
    assert 1 / np.mean(times) > 1500, f'FPS should be greater than 1500 but is {1 / np.mean(times)}'


if __name__ == '__main__':
    test_rl_env()
