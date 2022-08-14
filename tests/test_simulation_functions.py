# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test that all available environment calls work + check collisions are recorded correctly."""
import os

from hydra.core.global_hydra import GlobalHydra
from hydra import initialize, compose
import matplotlib.pyplot as plt
import numpy as np

from cfgs.config import PROJECT_PATH, get_scenario_dict
from nocturne import Simulation


def test_scenario_functions():
    """Check that collisions are appropriately recorded and that functions don't error."""
    GlobalHydra.instance().clear()
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    file_path = str(PROJECT_PATH / 'tests/large_file_tfrecord.json')
    os.environ["DISPLAY"] = ":0.0"
    ################################
    # Vehicle Collision checking
    ################################
    # now lets test for collisions
    # grab a vehicle and place it on top of another vehicle
    sim = Simulation(scenario_path=file_path, config=get_scenario_dict(cfg))
    scenario = sim.getScenario()
    veh0 = scenario.getVehicles()[0]
    veh1 = scenario.getVehicles()[1]
    veh2 = scenario.getVehicles()[2]
    # TODO(ev this fails unless the shift is non-zero)
    veh1.setPosition(veh0.getPosition().x + 0.001, veh0.getPosition().y)
    sim.step(0.000001)
    assert veh1.getCollided(
    ), 'vehicle1 should have collided after being placed on vehicle 0'
    assert veh0.getCollided(
    ), 'vehicle0 should have collided after vehicle 0 was placed on it'
    assert not veh2.getCollided(), 'vehicle2 should not have collided'

    # confirm that this is still true a time-step later
    sim.step(0.000001)
    assert veh1.getCollided(
    ), 'vehicle1 should have collided after being placed on vehicle 0'
    assert veh0.getCollided(
    ), 'vehicle0 should have collided after vehicle 0 was placed on it'
    assert not veh2.getCollided(), 'vehicle2 should not have collided'

    # now offset them slightly and do the same thing again
    sim = Simulation(scenario_path=file_path, config=get_scenario_dict(cfg))
    scenario = sim.getScenario()
    veh0 = scenario.getVehicles()[0]
    veh1 = scenario.getVehicles()[1]
    veh2 = scenario.getVehicles()[2]
    veh0 = scenario.getVehicles()[0]
    veh1 = scenario.getVehicles()[1]
    veh1.setPosition(veh0.getPosition().x + 0.2, veh0.getPosition().y + 0.2)
    sim.step(0.000001)
    assert veh1.getCollided(
    ), 'vehicle1 should have collided after being placed overlapping vehicle 0'
    assert veh0.getCollided(
    ), 'vehicle0 should have collided after vehicle 1 was placed on it'
    assert not veh2.getCollided(), 'vehicle2 should not have collided'

    ################################
    # Road Collision checking
    ################################
    # check if we place it onto one of the road points that there should be a collision
    print('entering road line - vehicle collision checking')
    # find a road edge
    colliding_road_line = None
    for roadline in scenario.getRoadLines():
        if roadline.canCollide():
            colliding_road_line = roadline
            break
    roadpoints = colliding_road_line.getGeometry()
    start_point = np.array([roadpoints[0].x, roadpoints[0].y])
    road_segment_dir = np.array([roadpoints[1].x, roadpoints[1].y]) - np.array(
        [roadpoints[0].x, roadpoints[0].y])
    assert np.linalg.norm(
        road_segment_dir) < 1  # it should be able to fit inside the vehicle
    road_segment_angle = np.arctan2(
        road_segment_dir[1], road_segment_dir[0])  # atan2 is (y, x) not (x,y)
    veh0.setHeading(road_segment_angle)

    # place the vehicle so that the segment is contained inside of it
    new_center = start_point + 0.5 * road_segment_dir
    veh0.setPosition(new_center[0], new_center[1])
    sim.step(1e-6)
    cone = scenario.getConeImage(veh0, view_angle=2 * np.pi, head_angle=0.0)
    plt.figure()
    plt.imshow(cone)
    plt.savefig('line_veh_check.png')
    assert veh0.getCollided(
    ), 'vehicle0 should have collided after a road edge is placed inside it'

    # place the vehicle on one of the points so that the road segment intersects with a vehicle edge
    sim.reset()
    scenario = sim.getScenario()
    veh0 = scenario.getVehicles()[0]
    veh0.setHeading(road_segment_angle)
    veh_length = veh0.length
    new_center += veh_length / 2 * road_segment_dir
    veh0.setPosition(new_center[0], new_center[1])
    sim.step(1e-6)
    assert veh0.getCollided(
    ), 'vehicle0 should have collided since a road edge intersects it'

    ######################
    # Waymo Scene Construction
    ######################
    # check that initializing things to a different time leads to a different
    # image
    cfg['scenario'].update({'start_time': 20})
    sim = Simulation(scenario_path=file_path, config=get_scenario_dict(cfg))
    scenario = sim.getScenario()

    img1 = scenario.getConeImage(scenario.getVehicles()[4], 120.0, 2 * np.pi,
                                 0.0)

    # check that initializing things with and without pedestrians leads to a different
    # image
    cfg['scenario'].update({'start_time': 20, 'allow_non_vehicles': False})
    sim = Simulation(scenario_path=file_path, config=get_scenario_dict(cfg))
    scenario = sim.getScenario()

    img2 = scenario.getConeImage(scenario.getVehicles()[4], 120.0, 2 * np.pi,
                                 0.0)
    assert not np.isclose(np.sum(img1 - img2),
                          0.0), 'adding pedestrians should change the image'

    # check a variety of nocturne functions
    _ = scenario.getPedestrians()
    _ = scenario.getCyclists()

    # check that the padding function for visible state is returning the right thing.
    visible_dict = scenario.visible_state(object=scenario.getVehicles()[0],
                                          view_dist=80,
                                          view_angle=120 * (np.pi / 180),
                                          padding=True)
    scenario_cfg = cfg['scenario']
    assert scenario_cfg['max_visible_objects'] == visible_dict['objects'].shape[0], \
        'visible dict padding returned {} objects but should have been \
            {}'.format(visible_dict['objects'].shape[0], scenario_cfg['max_visible_objects'])
    assert scenario_cfg['max_visible_road_points'] == visible_dict['road_points'].shape[0], \
        'visible dict padding returned {} objects but should have been \
            {}'.format(visible_dict['road_points'].shape[0], scenario_cfg['max_visible_road_points'])
    assert scenario_cfg['max_visible_traffic_lights'] == visible_dict['traffic_lights'].shape[0], \
        'visible dict padding returned {} objects but should have been \
            {}'.format(visible_dict['traffic_lights'].shape[0], scenario_cfg['max_visible_traffic_lights'])
    assert scenario_cfg['max_visible_stop_signs'] == visible_dict['stop_signs'].shape[0], \
        'visible dict padding returned {} objects but should have been \
            {}'.format(visible_dict['stop_signs'].shape[0], scenario_cfg['max_visible_stop_signs'])


def main():
    """See file docstring."""
    test_scenario_functions()


if __name__ == '__main__':
    main()
