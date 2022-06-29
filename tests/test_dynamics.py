# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test expert action computation from inverse dynamics."""
from hydra.core.global_hydra import GlobalHydra
from hydra import compose, initialize
import numpy as np

from cfgs.config import PROJECT_PATH, get_scenario_dict
from nocturne import Simulation

SIM_N_STEPS = 90  # number of steps per trajectory
SIM_STEP_TIME = 0.1  # dt (in seconds)


def _create_sim(file_path, expert_control):
    # None in the config would cause a bug
    GlobalHydra.instance().clear()
    initialize(config_path="../cfgs/")
    cfg = compose(config_name="config")
    # create simulation
    sim = Simulation(scenario_path=file_path, config=get_scenario_dict(cfg))
    # get controlled objects
    objects_that_moved = sim.getScenario().getObjectsThatMoved()
    for obj in objects_that_moved:
        obj.expert_control = expert_control
    return sim, objects_that_moved


def test_inverse_dynamics():
    """Check that expert actions are computed correctly from inverse dynamics."""
    file_path = str(PROJECT_PATH / 'tests/large_file_tfrecord.json')

    # create a ground truth sim that will replay expert actions
    sim_ground_truth, objects_ground_truth = _create_sim(file_path,
                                                         expert_control=True)
    id2obj_ground_truth = {obj.id: obj for obj in objects_ground_truth}
    # create a test sim that will replay actions from inverse dynamics
    sim_test, objects_test = _create_sim(file_path, expert_control=False)
    scenario_test = sim_test.getScenario()

    # step simulation
    for time in range(SIM_N_STEPS):
        # set model actions
        for obj in objects_test:
            action = scenario_test.expert_action(obj, time)
            if action is not None and not np.isnan(action.numpy()).any():
                # set object action
                obj.expert_control = False
                obj.acceleration = action.acceleration
                obj.steering = action.steering
            else:
                # set expert control for one time step
                obj.expert_control = True

        # step simulations
        sim_ground_truth.step(SIM_STEP_TIME)
        sim_test.step(SIM_STEP_TIME)

        # check positions
        for obj_test in objects_test:
            # only consider objects that used inverse dynamics action
            if obj_test.expert_control:
                continue
            # get corresponding ground truth object
            obj_ground_truth = id2obj_ground_truth[obj_test.id]

            # check that speeds and headings match
            assert np.isclose(obj_test.speed, obj_ground_truth.speed)
            assert np.isclose(obj_test.heading, obj_ground_truth.heading)

            # reposition objects
            obj_test.position = obj_ground_truth.position
            obj_test.heading = obj_ground_truth.heading
            obj_test.speed = obj_ground_truth.speed


if __name__ == '__main__':
    test_inverse_dynamics()
