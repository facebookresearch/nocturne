# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Make a movie from a random file."""
import os

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROCESSED_TRAIN_NO_TL, get_scenario_dict
from nocturne import Simulation


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """See file docstring."""
    disp = Display()
    disp.start()
    _ = plt.figure()
    files = os.listdir(PROCESSED_TRAIN_NO_TL)
    file = os.path.join(
        PROCESSED_TRAIN_NO_TL,
        files[np.random.randint(len(files))]
    )
    sim = Simulation(file, get_scenario_dict(cfg))
    frames = []
    scenario = sim.getScenario()
    for veh in scenario.getVehicles():
        veh.expert_control = True
    for i in range(90):
        img = scenario.getImage(
            img_width=1600,
            img_height=1600,
            draw_target_positions=False,
            padding=50.0,
        )
        frames.append(img)
        sim.step(0.1)

    movie_frames = np.array(frames)
    output_path = f'{os.path.basename(file)}.mp4'
    imageio.mimwrite(output_path, movie_frames, fps=30)
    import ipdb
    ipdb.set_trace()
    print('>', output_path)


if __name__ == '__main__':
    main()
