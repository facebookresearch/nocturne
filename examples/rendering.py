# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example of how to make movies of Nocturne scenarios."""
import os

import hydra
import imageio
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROJECT_PATH, get_scenario_dict
from nocturne import Simulation


def get_sim(cfg):
    """Initialize the scenario."""
    # load scenario, set vehicles to be expert-controlled
    sim = Simulation(scenario_path=str(PROJECT_PATH / 'examples' /
                                       'example_scenario.json'),
                     config=get_scenario_dict(cfg))
    for obj in sim.getScenario().getObjectsThatMoved():
        obj.expert_control = True
    return sim


def make_movie(cfg,
               scenario_fn,
               output_path='./vid.mp4',
               dt=0.1,
               steps=90,
               fps=10):
    """Make a movie from the scenario."""
    sim = get_sim(cfg)
    scenario = sim.getScenario()
    movie_frames = []
    timestep = 0
    movie_frames.append(scenario_fn(scenario, timestep))
    for i in range(steps):
        sim.step(dt)
        timestep += 1
        movie_frames.append(scenario_fn(scenario, timestep))
    movie_frames = np.array(movie_frames)
    imageio.mimwrite(output_path, movie_frames, fps=fps)
    print('>', output_path)
    del sim
    del movie_frames


def make_image(cfg, scenario_fn, output_path='./img.png'):
    """Make a single image from the scenario."""
    sim = get_sim(cfg)
    scenario = sim.getScenario()
    img = scenario_fn(scenario)
    dpi = 100
    height, width, depth = img.shape
    figsize = width / float(dpi), height / float(dpi)
    plt.figure(figsize=figsize, dpi=dpi)
    plt.axis('off')
    plt.imshow(img)
    plt.savefig(output_path)
    print('>', output_path)


@hydra.main(config_path="../cfgs/", config_name="config")
def main(cfg):
    """See file docstring."""
    # NOTE: don't run this file all at once since the memory usage for
    # rendering all the videos will be dozens of gigabytes
    disp = Display()
    disp.start()

    if not os.path.exists(PROJECT_PATH / 'examples/rendering'):
        os.makedirs(PROJECT_PATH / 'examples/rendering')

    # movie of whole scenario
    make_movie(
        cfg,
        scenario_fn=lambda scenario, _: scenario.getImage(
            img_width=1600,
            img_height=1600,
            draw_target_positions=True,
            padding=50.0,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' /
        'movie_whole_scenario.mp4',
    )

    # movie around a vehicle
    make_movie(
        cfg,
        scenario_fn=lambda scenario, _: scenario.getImage(
            img_width=1600,
            img_height=1600,
            draw_target_positions=True,
            padding=50.0,
            source=scenario.getVehicles()[3],
            view_width=120,
            view_height=120,
            rotate_with_source=True,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' /
        'movie_around_vehicle.mp4',
    )

    # movie around a vehicle (without rotating with source)
    make_movie(
        cfg,
        scenario_fn=lambda scenario, _: scenario.getImage(
            img_width=1600,
            img_height=1600,
            draw_target_positions=True,
            padding=50.0,
            source=scenario.getObjectsThatMoved()[0],
            view_width=120,
            view_height=120,
            rotate_with_source=False,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' /
        'movie_around_vehicle_stable.mp4',
    )

    # movie of cone around vehicle
    make_movie(
        cfg,
        scenario_fn=lambda scenario, _: scenario.getConeImage(
            source=scenario.getObjectsThatMoved()[0],
            view_dist=80,
            view_angle=np.pi * (120 / 180),
            head_angle=0.0,
            img_width=1600,
            img_height=1600,
            padding=50.0,
            draw_target_position=True,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' / 'movie_cone.mp4',
    )

    # movie of cone around vehicle with varying head angle
    make_movie(
        cfg,
        scenario_fn=lambda scenario, timestep: scenario.getConeImage(
            source=scenario.getVehicles()[6],
            view_dist=80.0,
            view_angle=np.pi * (120 / 180),
            head_angle=0.8 * np.sin(timestep / 10),
            img_width=1600,
            img_height=1600,
            padding=50.0,
            draw_target_position=True,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' /
        'movie_cone_head_angle.mp4',
    )

    # image of whole scenario
    make_image(
        cfg,
        scenario_fn=lambda scenario: scenario.getImage(
            img_width=2000,
            img_height=2000,
            padding=50.0,
            draw_target_positions=True,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' / 'img_scenario.png',
    )

    # image of cone
    make_image(
        cfg,
        scenario_fn=lambda scenario: scenario.getConeImage(
            source=scenario.getVehicles()[9],
            view_dist=80,
            view_angle=np.pi * (120 / 180),
            head_angle=np.pi / 8.0,
            img_width=2000,
            img_height=2000,
            padding=50.0,
            draw_target_position=True,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' /
        'img_cone_tilted.png',
    )

    # image of visible state
    make_image(
        cfg,
        scenario_fn=lambda scenario: scenario.getFeaturesImage(
            source=scenario.getVehicles()[9],
            view_dist=80,
            view_angle=np.pi * (120 / 180),
            head_angle=np.pi / 8.0,
            img_width=2000,
            img_height=2000,
            padding=50.0,
            draw_target_position=True,
        ),
        output_path=PROJECT_PATH / 'examples/rendering' /
        'img_features_tilted.png',
    )


if __name__ == '__main__':
    main()
