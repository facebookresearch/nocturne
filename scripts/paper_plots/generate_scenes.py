# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example of how to make movies of Nocturne scenarios."""
import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROJECT_PATH, \
    get_scenario_dict, set_display_window
from nocturne import Simulation


def get_sim(scenario_file, cfg):
    """Initialize the scenario."""
    # load scenario, set vehicles to be expert-controlled
    cfg['scenario']['allow_non_vehicles'] = False
    sim = Simulation(scenario_path=str(scenario_file),
                     config=get_scenario_dict(cfg))
    for obj in sim.getScenario().getObjectsThatMoved():
        obj.expert_control = True
    return sim


def make_movie(sim,
               scenario_fn,
               output_path='./vid.mp4',
               dt=0.1,
               steps=90,
               fps=10):
    """Make a movie from the scenario."""
    scenario = sim.getScenario()
    movie_frames = []
    timestep = 0
    movie_frames.append(scenario_fn(scenario, timestep))
    for i in range(steps):
        sim.step(dt)
        timestep += 1
        movie_frames.append(scenario_fn(scenario, timestep))
    movie_frames = np.stack(movie_frames, axis=0)
    imageio.mimwrite(output_path, movie_frames, fps=fps)
    print('>', output_path)
    del sim
    del movie_frames


def make_image(sim, scenario_file, scenario_fn, output_path='./img.png'):
    """Make a single image from the scenario."""
    scenario = sim.getScenario()
    img = scenario_fn(scenario)
    dpi = 100
    height, width, depth = img.shape
    figsize = width / dpi, height / dpi
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    print('>', output_path)


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """See file docstring."""
    set_display_window()

    # files = ['tfrecord-00358-of-01000_{}.json'.format(i) for i in range(500)]

    files = [
        'tfrecord-00358-of-01000_60.json',  # unprotected turn
        'tfrecord-00358-of-01000_72.json',  # four way stop
        'tfrecord-00358-of-01000_257.json',  # crowded four way stop
        'tfrecord-00358-of-01000_332.json',  # crowded merge road
        'tfrecord-00358-of-01000_79.json',  # crowded parking lot
    ]
    for file in files:
        file = os.path.join(PROCESSED_TRAIN_NO_TL, file)
        sim = get_sim(file, cfg)
        if os.path.exists(file):
            # image of whole scenario
            # make_image(
            #     sim,
            #     file,
            #     scenario_fn=lambda scenario: scenario.getImage(
            #         img_width=2000,
            #         img_height=2000,
            #         padding=50.0,
            #         draw_target_positions=True,
            #     ),
            #     output_path=PROJECT_PATH /
            #     'scripts/paper_plots/figs/scene_{}.png'.format(
            #         os.path.basename(file)),
            # )

            veh_index = -3
            make_image(
                sim,
                file,
                scenario_fn=lambda scenario: scenario.getImage(
                    img_height=1600,
                    img_width=1600,
                    draw_target_positions=True,
                    padding=0.0,
                    source=scenario.getVehicles()[veh_index],
                    view_height=80,
                    view_width=80,
                    rotate_with_source=True,
                ),
                output_path=PROJECT_PATH /
                'scripts/paper_plots/figs/cone_original_{}.png'.format(
                    os.path.basename(file)),
            )
            make_image(
                sim,
                file,
                scenario_fn=lambda scenario: scenario.getConeImage(
                    source=scenario.getVehicles()[veh_index],
                    view_dist=cfg['subscriber']['view_dist'],
                    view_angle=cfg['subscriber']['view_angle'],
                    head_angle=0.0,
                    img_height=1600,
                    img_width=1600,
                    padding=0.0,
                    draw_target_position=True,
                ),
                output_path=PROJECT_PATH /
                'scripts/paper_plots/figs/cone_{}.png'.format(
                    os.path.basename(file)),
            )
            make_image(
                sim,
                file,
                scenario_fn=lambda scenario: scenario.getFeaturesImage(
                    source=scenario.getVehicles()[veh_index],
                    view_dist=cfg['subscriber']['view_dist'],
                    view_angle=cfg['subscriber']['view_angle'],
                    head_angle=0.0,
                    img_height=1600,
                    img_width=1600,
                    padding=0.0,
                    draw_target_position=True,
                ),
                output_path=PROJECT_PATH /
                'scripts/paper_plots/figs/feature_{}.png'.format(
                    os.path.basename(file)),
            )


if __name__ == '__main__':
    main()
