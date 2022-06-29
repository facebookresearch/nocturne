# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Test step and rendering functions."""
import hydra

from cfgs.config import set_display_window
from nocturne import Action
from nocturne.envs.wrappers import create_env


@hydra.main(config_path="../cfgs/", config_name="config")
def create_rl_env(cfg):
    """Test step and rendering functions."""
    set_display_window()
    env = create_env(cfg)
    _ = env.reset()
    # quick check that rendering works
    _ = env.scenario.getConeImage(
        env.scenario.getVehicles()[0],
        # how far the agent can see
        view_dist=cfg['subscriber']['view_dist'],
        # the angle formed by the view cone
        view_angle=cfg['subscriber']['view_angle'],
        # the agent's head angle
        head_angle=0.0,
        # whether to draw the goal position in the image
        draw_target_position=False)
    for _ in range(80):
        # grab the list of vehicles that actually need to
        # move some distance to get to their goal
        moving_vehs = env.scenario.getObjectsThatMoved()
        # obs, rew, done, info
        # each of these objects is a dict keyed by the vehicle ID
        # info[veh_id] contains the following useful keys:
        # 'collided': did the agent collide with a road object or edge
        # 'veh_veh_collision': did the agent collide with a vehicle
        # 'veh_edge_collision': did the agent collide with a road edge
        # 'goal_achieved': did we get to our target
        _, _, _, _ = env.step({
            veh.id: Action(acceleration=2.0, steering=1.0, head_angle=0.5)
            for veh in moving_vehs
        })


if __name__ == '__main__':
    create_rl_env()
