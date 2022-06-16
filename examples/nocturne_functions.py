# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Example of how to make movies of Nocturne scenarios."""
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display

from cfgs.config import PROJECT_PATH, get_scenario_dict
from nocturne import Simulation, Action


def save_image(img, output_path='./img.png'):
    """Make a single image from the scenario."""
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
    """Initialize the scenario."""
    disp = Display()
    disp.start()
    if not os.path.exists(PROJECT_PATH / 'examples/rendering'):
        os.makedirs(PROJECT_PATH / 'examples/rendering')
    # load scenario. by default this won't have pedestrians or cyclists
    sim = Simulation(scenario_path=str(PROJECT_PATH / 'examples' /
                                       'example_scenario.json'),
                     config=get_scenario_dict(cfg))
    scenario = sim.getScenario()
    img = scenario.getImage(
        img_width=2000,
        img_height=2000,
        padding=50.0,
        draw_target_positions=True,
    )
    save_image(img,
               PROJECT_PATH / 'examples/rendering' / 'scene_with_no_peds.png')
    # grab all the vehicles
    vehs = scenario.getVehicles()
    # grab all the vehicles that moved and show some things
    # we can do with them
    vehs = scenario.getObjectsThatMoved()
    vehs[0].highlight = True  # draw a circle around it on the rendered image
    # setting a vehicle to expert_control will cause
    # this agent will replay expert data starting frmo
    # the current time in the simulation
    vehs[0].expert_control = True
    print(f'width is {vehs[0].width}, length is {vehs[0].length}')
    print(f'speed is {vehs[0].speed}, heading is {vehs[0].heading}')
    print(f'position is {vehs[0].width}, length is {vehs[0].length}')
    # for efficiency, we return position as a custom Vector2D object
    # this object can be converted to and from numpy and comes with
    # support for a variety of algebraic operations
    print(f'position is {vehs[0].position}')
    print(f'position as numpy array is {vehs[0].position.numpy()}')
    print(f'norm of position is {vehs[0].position.norm()}')
    print(f'angle in a world-centered frame {vehs[0].position.angle()}')
    print(f'rotated position is {vehs[0].position.rotate(np.pi).numpy()}')
    # we can set vehicle accel, steering, head angle directly
    vehs[0].acceleration = -1
    vehs[0].steering = 1
    vehs[0].head_angle = np.pi
    # we can also set them all directly using an action object
    vehs[0].apply_action(Action(acceleration=-1, steering=1, head_angle=np.pi))
    # we can grab the state for this vehicle in two way:
    # 1) a flattened vector corresponding to the set of visible objects
    # concatenated according to [visible objects, visible road points,
    #                           visible stop signs, visible traffic lights]
    # note that since we want to make a fixed length vector, for each of these
    # types the config, under the scenario key has the following items
    # max_visible_objects: 16
    # max_visible_road_points: 1000
    # max_visible_traffic_lights: 20
    # max_visible_stop_signs: 4
    # we grab all the visible items for each type, sort them by distance from
    # the vehicle and return the closest. If we have fewer than the maximum
    # we pad with 0s.
    flattened_vector = scenario.flattened_visible_state(object=vehs[0],
                                                        view_dist=80,
                                                        view_angle=120 *
                                                        (np.pi / 180),
                                                        head_angle=0.0)
    # we can also grab a dict of all of the objects
    # if padding is true we will add extra objects to the dict
    # to ensure we hit the maximum number of objects for each type
    visible_dict = scenario.visible_state(object=vehs[0],
                                          view_dist=80,
                                          view_angle=120 * (np.pi / 180),
                                          padding=False)

    # load scenario, this time with pedestrians and cyclists
    cfg['scenario']['allow_non_vehicles'] = True
    sim = Simulation(scenario_path=str(PROJECT_PATH / 'examples' /
                                       'example_scenario.json'),
                     config=get_scenario_dict(cfg))
    scenario = sim.getScenario()
    img = scenario.getImage(
        img_width=2000,
        img_height=2000,
        padding=50.0,
        draw_target_positions=True,
    )
    save_image(img,
               PROJECT_PATH / 'examples/rendering' / 'scene_with_peds.png')
    # now we need to be slightly more careful about how we select objects
    # since getMovingObjects will return pedestrians and cyclists
    # and getVehicles will return vehicles that don't necessarily need to move
    objects_that_moved = scenario.getObjectsThatMoved()
    objects_of_interest = [
        obj for obj in scenario.getVehicles() if obj in objects_that_moved
    ]  # noqa: 841
    vehicles = scenario.getVehicles()
    cyclists = scenario.getCyclists()
    pedestrians = scenario.getPedestrians()
    all_objects = scenario.getObjects()


if __name__ == '__main__':
    main()
