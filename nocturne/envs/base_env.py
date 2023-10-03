# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Default Nocturne env with minor adaptations."""

import json
import logging
import os
from collections import defaultdict, deque
from itertools import islice
from typing import Any, Dict, Sequence, Union

import numpy as np
import torch
from gym import Env
from gym.spaces import Box, Discrete

from nocturne import Action, Simulation

logging.getLogger(__name__)


class BaseEnv(Env):
    def __init__(self, config: Dict[str, Any], rank: int = 0) -> None:
        """Initialize a Nocturne environment.

        Args
        ----
            config (dict): configuration file for the environment.
            data_path (str): path to the data directory with traffic scenes.
            rank (int, optional): [description]. Defaults to 0.
        """
        super().__init__()
        self.config = config

        # Path to traffic scene(s) to use
        self._data_path = self.config["data_path"]

        # Load the list of valid files
        with open(os.path.join(self._data_path, "valid_files.json")) as file:
            self.valid_veh_dict = json.load(file)
            self.files = list(self.valid_veh_dict.keys())
            # sort the files so that we have a consistent order
            self.files = sorted(self.files)
        if self.config["num_files"] != -1:
            self.files = self.files[: self.config["num_files"]]
        self.file = (
            self.config["scene"]
            if "scene" in self.config and self.config["scene"] is not None
            else self.files[np.random.randint(len(self.files))]
        )
        self.simulation = Simulation(
            os.path.join(self._data_path, self.file),
            config=self.config["scenario"],
        )
        self.scenario = self.simulation.getScenario()
        self.controlled_vehicles = self.scenario.getObjectsThatMoved()
        self._invalid_position = float(-1e4)

        self.start_positions = {
            veh_obj.id: np.array([veh_obj.position.x, veh_obj.position.y]) for veh_obj in self.controlled_vehicles
        }
        self.n_frames_stacked = self.config["subscriber"].get("n_frames_stacked", 1)
        if self.n_frames_stacked > 1:
            logging.warning("Frame stacking is enabled. Note that this is not required for " "recurrent policies.")
        self.max_num_vehicles = self.config["max_num_vehicles"]
        self.single_agent_mode = self.config["single_agent_mode"]
        if self.single_agent_mode:
            self.max_num_vehicles = 1
        self.seed(self.config["seed"])
        self.episode_length = self.config["episode_length"]
        self.t = 0
        self.step_num = 0
        self.rank = rank
        obs_dict = self.reset()
        self.observation_space = Box(
            low=-np.infty,
            high=np.infty,
            shape=(obs_dict[list(obs_dict.keys())[0]].shape[0],),
        )
        if self.config["discretize_actions"]:
            self.accel_discretization = self.config["accel_discretization"]
            self.steering_discretization = self.config["steering_discretization"]
            self.action_space = Discrete(self.accel_discretization * self.steering_discretization)
            self.accel_grid = np.linspace(
                -np.abs(self.config["accel_lower_bound"]),
                self.config["accel_upper_bound"],
                self.accel_discretization,
            )
            self.steering_grid = np.linspace(
                -np.abs(self.config["steering_lower_bound"]),
                self.config["steering_upper_bound"],
                self.steering_discretization,
            )

            # compute the indexing only once
            self.idx_to_actions = {}
            i = 0
            for accel in self.accel_grid:
                for steer in self.steering_grid:
                    self.idx_to_actions[i] = [accel, steer]
                    i += 1
        else:
            self.action_space = Box(
                low=-np.array(
                    [
                        np.abs(self.config["accel_lower_bound"]),
                        self.config["steering_lower_bound"],
                    ]
                ),
                high=np.array(
                    [
                        np.abs(self.config["accel_upper_bound"]),
                        self.config["steering_upper_bound"],
                    ]
                ),
            )

    def apply_actions(self, action_dict: Dict[int, Union[Action, np.ndarray, Sequence[float], int]]) -> None:
        """Apply a dict of actions to the vehicle objects."""
        for veh_obj in self.scenario.getObjectsThatMoved():
            action = action_dict.get(veh_obj.id, None)
            if action is None:
                continue

            if isinstance(action, Action):
                veh_obj.apply_action(action)
            elif isinstance(action, np.ndarray):
                veh_obj.apply_action(Action.from_numpy(action))
            elif isinstance(action, (tuple, list)):
                veh_obj.acceleration = action[0]
                veh_obj.steering = action[1]
            else:
                accel, steer = self.idx_to_actions[action]
                veh_obj.acceleration = accel
                veh_obj.steering = steer

    def step(self, action_dict: Dict[int, Union[Action, np.ndarray, Sequence[float], int]]) -> None:
        """See superclass."""
        obs_dict = {}
        rew_dict = {}
        done_dict = {}
        info_dict = defaultdict(dict)
        rew_cfg = self.config["rew_cfg"]
        self.apply_actions(action_dict)
        self.simulation.step(self.config["dt"])
        self.t += self.config["dt"]
        self.step_num += 1
        objs_to_remove = []
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            if veh_id in self.done_ids:
                continue
            self.context_dict[veh_id].append(self.get_observation(veh_obj))
            if self.n_frames_stacked > 1:
                veh_deque = self.context_dict[veh_id]
                context_list = list(
                    islice(
                        veh_deque,
                        len(veh_deque) - self.n_frames_stacked,
                        len(veh_deque),
                    )
                )
                obs_dict[veh_id] = np.concatenate(context_list)
            else:
                obs_dict[veh_id] = self.context_dict[veh_id][-1]
            rew_dict[veh_id] = 0
            done_dict[veh_id] = False
            info_dict[veh_id]["goal_achieved"] = False
            info_dict[veh_id]["collided"] = False
            info_dict[veh_id]["veh_veh_collision"] = False
            info_dict[veh_id]["veh_edge_collision"] = False
            obj_pos = veh_obj.position
            goal_pos = veh_obj.target_position
            """############################################
                            Compute rewards
               ############################################"""
            position_target_achieved = True
            speed_target_achieved = True
            heading_target_achieved = True
            if rew_cfg["position_target"]:
                position_target_achieved = (goal_pos - obj_pos).norm() < rew_cfg["position_target_tolerance"]
            if rew_cfg["speed_target"]:
                speed_target_achieved = np.abs(veh_obj.speed - veh_obj.target_speed) < rew_cfg["speed_target_tolerance"]
            if rew_cfg["heading_target"]:
                heading_target_achieved = (
                    np.abs(self.angle_sub(veh_obj.heading, veh_obj.target_heading))
                    < rew_cfg["heading_target_tolerance"]
                )
            if position_target_achieved and speed_target_achieved and heading_target_achieved:
                info_dict[veh_id]["goal_achieved"] = True
                rew_dict[veh_id] += rew_cfg["goal_achieved_bonus"] / rew_cfg["reward_scaling"]
            if rew_cfg["shaped_goal_distance"] and rew_cfg["position_target"]:
                # penalize the agent for its distance from goal
                # we scale by goal_dist_normalizers to ensure that this value is always
                # less than the penalty for collision
                if rew_cfg["goal_distance_penalty"]:
                    rew_dict[veh_id] -= (
                        rew_cfg.get("shaped_goal_distance_scaling", 1.0)
                        * ((goal_pos - obj_pos).norm() / self.goal_dist_normalizers[veh_id])
                        / rew_cfg["reward_scaling"]
                    )
                else:
                    # the minus one is to ensure that it's not beneficial to collide
                    # we divide by goal_achieved_bonus / episode_length to ensure that
                    # acquiring the maximum "get-close-to-goal" reward at every
                    # time-step is always less than just acquiring the goal reward once
                    # we also assume that vehicles are never more than 400 meters from
                    # their goal which makes sense as the episodes are 9 seconds long
                    # i.e. we'd have to go more than 40 m/s to get there
                    rew_dict[veh_id] += (
                        rew_cfg.get("shaped_goal_distance_scaling", 1.0)
                        * (1 - (goal_pos - obj_pos).norm() / self.goal_dist_normalizers[veh_id])
                        / rew_cfg["reward_scaling"]
                    )
                # repeat the same thing for speed and heading
                if rew_cfg["shaped_goal_distance"] and rew_cfg["speed_target"]:
                    if rew_cfg["goal_distance_penalty"]:
                        rew_dict[veh_id] -= (
                            rew_cfg.get("shaped_goal_distance_scaling", 1.0)
                            * (np.abs(veh_obj.speed - veh_obj.target_speed) / 40.0)
                            / rew_cfg["reward_scaling"]
                        )
                    else:
                        rew_dict[veh_id] += (
                            rew_cfg.get("shaped_goal_distance_scaling", 1.0)
                            * (1 - np.abs(veh_obj.speed - veh_obj.target_speed) / 40.0)
                            / rew_cfg["reward_scaling"]
                        )
                if rew_cfg["shaped_goal_distance"] and rew_cfg["heading_target"]:
                    if rew_cfg["goal_distance_penalty"]:
                        rew_dict[veh_id] -= (
                            rew_cfg.get("shaped_goal_distance_scaling", 1.0)
                            * (np.abs(self.angle_sub(veh_obj.heading, veh_obj.target_heading)) / (2 * np.pi))
                            / rew_cfg["reward_scaling"]
                        )
                    else:
                        rew_dict[veh_id] += (
                            rew_cfg.get("shaped_goal_distance_scaling", 1.0)
                            * (1 - np.abs(self.angle_sub(veh_obj.heading, veh_obj.target_heading)) / (2 * np.pi))
                            / rew_cfg["reward_scaling"]
                        )
            """############################################
                    Handle potential done conditions
            ############################################"""
            # achieved our goal
            if info_dict[veh_id]["goal_achieved"] and self.config.get("remove_at_goal", True):
                done_dict[veh_id] = True
            if veh_obj.getCollided():
                info_dict[veh_id]["collided"] = True
                if int(veh_obj.collision_type) == 1:
                    info_dict[veh_id]["veh_veh_collision"] = True
                if int(veh_obj.collision_type) == 2:
                    info_dict[veh_id]["veh_edge_collision"] = True
                rew_dict[veh_id] -= np.abs(rew_cfg["collision_penalty"]) / rew_cfg["reward_scaling"]
                if self.config.get("remove_at_collide", True):
                    done_dict[veh_id] = True
            # remove the vehicle so that its trajectory doesn't continue. This is
            # important in the multi-agent setting.
            if done_dict[veh_id]:
                self.done_ids.append(veh_id)
                if (info_dict[veh_id]["goal_achieved"] and self.config.get("remove_at_goal", True)) or (
                    info_dict[veh_id]["collided"] and self.config.get("remove_at_collide", True)
                ):
                    objs_to_remove.append(veh_obj)

        for veh_obj in objs_to_remove:
            self.scenario.removeVehicle(veh_obj)

        if self.config["rew_cfg"]["shared_reward"]:
            total_reward = np.sum([rew_dict[key] for key in rew_dict.keys()])
            rew_dict = {key: total_reward for key in rew_dict.keys()}

        # fill in the missing observations if we should be doing so
        if self.config["subscriber"]["keep_inactive_agents"]:
            # force all vehicles done to be false since they should persist through the
            # episode
            done_dict = {key: False for key in self.all_vehicle_ids}
            for key in self.all_vehicle_ids:
                if key not in obs_dict.keys():
                    obs_dict[key] = self.dead_feat
                    rew_dict[key] = 0.0
                    info_dict[key]["goal_achieved"] = False
                    info_dict[key]["collided"] = False
                    info_dict[key]["veh_veh_collision"] = False
                    info_dict[key]["veh_edge_collision"] = False

        if self.step_num >= self.episode_length:
            done_dict = {key: True for key in done_dict.keys()}

        all_done = True
        for value in done_dict.values():
            all_done *= value
        done_dict["__all__"] = all_done

        return obs_dict, rew_dict, done_dict, info_dict

    def reset(self):
        """See superclass."""
        self.t = 0
        self.step_num = 0

        enough_vehicles = False
        # we don't want to initialize scenes with 0 actors after satisfying
        # all the conditions on a scene that we have
        while not enough_vehicles:
            self.file = (
                self.config["scene"]
                if "scene" in self.config and self.config["scene"] is not None
                else self.files[np.random.randint(len(self.files))]
            )
            self.simulation = Simulation(
                os.path.join(self._data_path, self.file),
                config=self.config["scenario"],
            )
            self.scenario = self.simulation.getScenario()
            """##################################################################
                Construct context dictionary of observations that can be used to
                warm up policies by stepping all vehicles as experts.
            #####################################################################"""
            dead_obs = self.get_observation(self.scenario.getVehicles()[0])
            self.dead_feat = -np.ones(dead_obs.shape[0] * self.n_frames_stacked)
            # step all the vehicles forward by one second and record their observations
            # as context
            context_len = max(10, self.n_frames_stacked)
            self.context_dict = {
                veh.getID(): deque([self.dead_feat for _ in range(context_len)], maxlen=context_len)
                for veh in self.scenario.getObjectsThatMoved()
            }
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = True
            for _ in range(10):
                for veh in self.scenario.getObjectsThatMoved():
                    self.context_dict[veh.getID()].append(self.get_observation(veh))
                self.simulation.step(self.config["dt"])
            # now hand back control to our actual controllers
            for veh in self.scenario.getObjectsThatMoved():
                veh.expert_control = False

            # remove all the objects that are in collision or are already in goal dist
            # additionally set the objects that have infeasible goals to be experts
            for veh_obj in self.simulation.getScenario().getObjectsThatMoved():
                obj_pos = veh_obj.getPosition()
                obj_pos = np.array([obj_pos.x, obj_pos.y])
                goal_pos = veh_obj.getGoalPosition()
                goal_pos = np.array([goal_pos.x, goal_pos.y])
                """############################################
                    Remove vehicles at goal
                ############################################"""
                norm = np.linalg.norm(goal_pos - obj_pos)
                if norm < self.config["rew_cfg"]["goal_tolerance"] or veh_obj.getCollided():
                    self.scenario.removeVehicle(veh_obj)
                """############################################
                    Set all vehicles with unachievable goals to be experts
                ############################################"""
                if self.file in self.valid_veh_dict and veh_obj.getID() in self.valid_veh_dict[self.file]:
                    veh_obj.expert_control = True
            """############################################
                Pick out the vehicles that we are controlling
            ############################################"""
            # ensure that we have no more than max_num_vehicles are controlled
            temp_vehicles = self.scenario.getObjectsThatMoved()
            np.random.shuffle(temp_vehicles)
            curr_index = 0
            self.controlled_vehicles = []
            self.expert_controlled_vehicles = []
            self.vehicles_to_delete = []
            for vehicle in temp_vehicles:
                # this vehicle was invalid at the end of the 1 second context
                # step so we need to remove it.
                if np.isclose(vehicle.position.x, self._invalid_position):
                    self.vehicles_to_delete.append(vehicle)
                # If vehicle ID is given, use that as controlled vehicle
                elif "vehicle" in self.config and self.config["vehicle"] is not None:
                    if vehicle.id == self.config["vehicle"]:
                        self.controlled_vehicles.append(vehicle)
                    else:
                        self.expert_controlled_vehicles.append(vehicle)
                # we don't want to include vehicles that had unachievable goals
                # as controlled vehicles
                elif not vehicle.expert_control and curr_index < self.max_num_vehicles:
                    self.controlled_vehicles.append(vehicle)
                    curr_index += 1
                else:
                    self.expert_controlled_vehicles.append(vehicle)
            self.all_vehicle_ids = [veh.getID() for veh in self.controlled_vehicles]
            # make all the vehicles that are in excess of max_num_vehicles controlled by
            # an expert
            for veh in self.expert_controlled_vehicles:
                veh.expert_control = True
            # remove vehicles that are currently at an invalid position
            for veh in self.vehicles_to_delete:
                self.scenario.removeVehicle(veh)

            # check that we have at least one vehicle or if we have just one file, exit
            # anyways
            # or else we might be stuck in an infinite loop
            if len(self.all_vehicle_ids) > 0 or (
                len(self.files) == 1 or ("scene" in self.config and self.config["scene"] is not None)
            ):
                enough_vehicles = True

        # for one reason or another (probably we had a file where all the agents
        # achieved their goals)
        # we have no controlled vehicles
        # just grab a vehicle even if it hasn't moved so that we have something
        # to return obs for even if it's not controlled
        # NOTE: this case only occurs during our eval procedure where we set the
        # self.files list to be length 1. Otherwise, the while loop above will repeat
        # until a file is found.
        if len(self.all_vehicle_ids) == 0:
            self.controlled_vehicles = [self.scenario.getVehicles()[0]]
            self.all_vehicle_ids = [veh.getID() for veh in self.controlled_vehicles]

        # construct the observations and goal normalizers
        obs_dict = {}
        self.goal_dist_normalizers = {}
        max_goal_dist = -100
        for veh_obj in self.controlled_vehicles:
            veh_id = veh_obj.getID()
            # store normalizers for each vehicle
            obj_pos = veh_obj.getPosition()
            obj_pos = np.array([obj_pos.x, obj_pos.y])
            goal_pos = veh_obj.getGoalPosition()
            goal_pos = np.array([goal_pos.x, goal_pos.y])
            dist = np.linalg.norm(obj_pos - goal_pos)
            self.goal_dist_normalizers[veh_id] = dist
            # compute the obs
            self.context_dict[veh_id].append(self.get_observation(veh_obj))
            if self.n_frames_stacked > 1:
                veh_deque = self.context_dict[veh_id]
                context_list = list(
                    islice(
                        veh_deque,
                        len(veh_deque) - self.n_frames_stacked,
                        len(veh_deque),
                    )
                )
                obs_dict[veh_id] = np.concatenate(context_list)
            else:
                obs_dict[veh_id] = self.context_dict[veh_id][-1]
            # pick the vehicle that has to travel the furthest distance and use it for
            # rendering
            if dist > max_goal_dist:
                # this attribute is just used for rendering of the view
                # from the ego frame
                self.render_vehicle = veh_obj
                max_goal_dist = dist

        self.done_ids = []
        # we should return obs for the missing agents
        if self.config["subscriber"]["keep_inactive_agents"]:
            max_id = max([int(key) for key in obs_dict.keys()])
            num_missing_agents = max(0, self.max_num_vehicles - len(obs_dict))
            for i in range(num_missing_agents):
                obs_dict[max_id + i + 1] = self.dead_feat
            self.initial_dead_agent_ids = [max_id + i + 1 for i in range(num_missing_agents)]
            self.all_vehicle_ids = list(obs_dict.keys())
        else:
            self.initial_dead_agent_ids = []

        logging.debug(f"Scene: {self.file} | Controlling vehicles: " f"{[veh.id for veh in self.controlled_vehicles]}")

        return obs_dict

    def get_observation(self, veh_obj):
        """Return the observation for a particular vehicle."""

        use_ego_state = self.config["subscriber"]["use_ego_state"]
        use_observations = self.config["subscriber"]["use_observations"]
        use_current_position = self.config["subscriber"]["use_current_position"]

        view_dist = self.config["subscriber"]["view_dist"]
        view_angle = self.config["subscriber"]["view_angle"]
        cur_pos = np.array([veh_obj.getPosition().x, veh_obj.getPosition().y])

        obs = np.concatenate(
            (
                self.scenario.ego_state(veh_obj) if use_ego_state else [],
                cur_pos if use_current_position else [],
                self.scenario.flattened_visible_state(veh_obj, view_dist, view_angle) if use_observations else [],
            )
        )
        return obs

    def make_all_vehicles_experts(self):
        """Force all vehicles to be experts."""
        for veh in self.scenario.getVehicles():
            veh.expert_control = True

    def get_vehicles(self):
        """Return the vehicles."""
        return self.scenario.getVehicles()

    def get_objects_that_moved(self):
        """Return the objects that moved."""
        return self.scenario.getObjectsThatMoved()

    def render(self, mode=None):
        """See superclass."""
        return self.scenario.getImage(
            img_width=1600,
            img_height=1600,
            draw_target_positions=True,
            padding=50.0,
        )

    def render_ego(self, mode=None):
        """See superclass."""
        if self.render_vehicle.getID() in self.done_ids:
            return None
        else:
            return self.scenario.getConeImage(
                source=self.render_vehicle,
                view_dist=self.config["subscriber"]["view_dist"],
                view_angle=self.config["subscriber"]["view_angle"],
                head_angle=self.render_vehicle.head_angle,
                img_width=1600,
                img_height=1600,
                padding=50.0,
                draw_target_position=True,
            )

    def render_features(self, mode=None):
        """See superclass."""
        if self.render_vehicle.getID() in self.done_ids:
            return None
        else:
            return self.scenario.getFeaturesImage(
                source=self.render_vehicle,
                view_dist=self.config["subscriber"]["view_dist"],
                view_angle=self.config["subscriber"]["view_angle"],
                head_angle=self.render_vehicle.head_angle,
                img_width=1600,
                img_height=1600,
                padding=50.0,
                draw_target_position=True,
            )

    def seed(self, seed=None):
        """Ensure determinism."""
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def angle_sub(self, current_angle, target_angle) -> int:
        """Subtract two angles to find the minimum angle between them."""
        # Subtract the angles, constraining the value to [0, 2 * np.pi)
        diff = (target_angle - current_angle) % (2 * np.pi)

        # If we are more than np.pi we're taking the long way around.
        # Let's instead go in the shorter, negative direction
        if diff > np.pi:
            diff = -(2 * np.pi - diff)
        return diff
