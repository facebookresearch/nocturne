import json
import logging
import numpy as np
import pandas as pd
from itertools import product

import gymnasium as gym
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader
from gym.spaces import Discrete

from utils.config import load_config
from nocturne import Simulation
from nocturne.envs.base_env import BaseEnv

# Global setting
logging.basicConfig(level="INFO")

class TrajectoryIterator(IterableDataset):
    """Generates trajectories in Waymo scenes: sequences of observations and actions."""

    def __init__(self, data_path, env_config, with_replacement=True, file_limit=-1):
        self.data_path = data_path
        self.config = env_config
        self.env = BaseEnv(env_config)
        self.with_replacement = with_replacement
        self.valid_veh_dict = json.load(open(f"{self.data_path}/valid_files.json", "r", encoding="utf-8"))
        self.file_names = sorted(list(self.valid_veh_dict.keys()))[:file_limit]
        self._set_discrete_action_space()
        self.action_space = gym.spaces.Discrete(len(self.actions_to_joint_idx))
        self.ep_norm_rewards = []

        super(TrajectoryIterator).__init__()

        logging.info(f"Using {len(self.file_names)} file(s)")
        
    def __iter__(self):
        """Return an (expert_state, expert_action) iterable."""
        return self._get_trajectories()

    def _get_trajectories(self):
        """Load scenes, preprocess and return trajectories."""

        if len(self.file_names) == None:
            logging.info("file_names is empty.")
            return None

        while True:

            # (1) Sample traffic scene
            if self.with_replacement:
                filename = np.random.choice(self.file_names)
            else: # Every scene can only be used once
                filename = self.file_names.pop()

            # (2) Obtain discretized expert actions
            expert_actions_df = self._discretize_expert_actions(filename)

            # (3) Obtain corrected observations
            expert_obs, expert_acts, expert_next_obs, expert_dones = self._step_through_scene(expert_actions_df, filename)

            # (4) Return
            for obs, act, next_obs, done in zip(expert_obs, expert_acts, expert_next_obs, expert_dones):
                yield (obs, act, next_obs, done)

    def _discretize_expert_actions(self, filename: str):
        """Discretize human expert actions in given traffic scene."""

        # Create simulation
        sim = Simulation(f"{self.data_path}/{filename}", dict(self.config.scenario))
        scenario = sim.getScenario()

        # Set expert-controlled to False
        for obj in scenario.getObjects():
            obj.expert_control = True

        objects_that_moved = scenario.getObjectsThatMoved()
        objects_of_interest = [
            obj for obj in scenario.getVehicles() if obj in objects_that_moved
            and obj.getID() not in self.valid_veh_dict[filename]
        ]

        # Setup dataframe to store actions
        actions_dict = {}
        for agent in objects_of_interest:
            actions_dict[agent.id] = np.zeros(self.config.episode_length)

        df_actions = pd.DataFrame(actions_dict)

        for timestep in range(self.config.episode_length + self.config.warmup_period):
            for veh_obj in objects_of_interest:
            
                # Get (continuous) expert action
                expert_action = scenario.expert_action(veh_obj, timestep)

                # Check for invalid actions (None) (because no value available for taking
                # derivative) or because the vehicle is at an invalid state
                if expert_action is None:
                    continue

                expert_accel, expert_steering, _ = expert_action.numpy()

                # Map actions to nearest grid indices and joint action 
                accel_grid_val, accel_grid_idx = self._find_closest_index(self.accel_grid, expert_accel)
                steering_grid_val, steering_grid_idx = self._find_closest_index(self.steering_grid, expert_steering)

                expert_action_idx = self.actions_to_joint_idx[accel_grid_val, steering_grid_val][0]
                
                if expert_action_idx is None:
                    logging.debug("Expert action is None!")
                
                # Store
                if timestep >= self.config.warmup_period:
                    df_actions.loc[timestep-self.config.warmup_period][veh_obj.getID()] = expert_action_idx

            sim.step(self.config.dt)

        return df_actions
    
    def _step_through_scene(self, expert_actions_df: pd.DataFrame, filename: str):
        """
        Step through a traffic scenario using a set of discretized expert actions 
        to construct a set of corrected state-action pairs. Note: A state-action pair 
        is the observation + the action chosen given that observation.
        """
        # Make and reset environment
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, self.env.observation_space.shape, np.float32
        )
        
        # Reset
        next_obs_dict = self.env.reset(filename)
        num_agents = expert_actions_df.shape[1]

        # Storage
        expert_action_arr = np.full((expert_actions_df.shape), fill_value=np.nan)
        obs_arr = np.full(
            (
                expert_actions_df.shape[0],
                num_agents,
                self.env.observation_space.shape[0],
            ), fill_value=np.nan
        )
        next_obs_arr = np.full_like(obs_arr, fill_value=np.nan)
        dones_arr = np.full_like(expert_action_arr, fill_value=np.nan)
        
        ep_rewards = np.zeros(num_agents)
        dead_agent_ids = []

        # Select agents of interest
        agents_of_interest = self.env.controlled_vehicles

        # (TODO: Add option to step through in expert controlled mode)
        for agent in agents_of_interest:
            agent.expert_control = False

        for timestep in range(self.config.episode_length):

            # Select action from expert grid actions dataframe
            action_dict = {}
            for agent_idx, agent in enumerate(agents_of_interest):
                if agent.id in next_obs_dict:
                    action = int(expert_actions_df[agent.id].loc[timestep])
                    action_dict[agent.id] = action

            # Store actions + obervations of living agents
            for agent_idx, agent in enumerate(agents_of_interest):
                if agent.id not in dead_agent_ids:
                    obs_arr[timestep, agent_idx, :] = next_obs_dict[agent.id]
                    expert_action_arr[timestep, agent_idx] = action_dict[agent.id]

            # Execute actions
            next_obs_dict, rew_dict, done_dict, info_dict = self.env.step(action_dict)

            # The i'th observation `next_obs[i]` in this array is the observation
            # after the agent has taken action `acts[i]`.
            for agent_idx, agent in enumerate(agents_of_interest):
                if agent.id not in dead_agent_ids:
                    next_obs_arr[timestep, agent_idx, :] = next_obs_dict[agent.id]
                    dones_arr[timestep, agent_idx] = done_dict[agent.id]

            # Update rewards
            for agent_idx, agent in enumerate(agents_of_interest):
                if agent.id in rew_dict:
                    ep_rewards[agent_idx] += rew_dict[agent.id]

            # Update dead agents
            for agent_id, is_done in done_dict.items():
                if is_done and agent_id not in dead_agent_ids:
                    dead_agent_ids.append(agent_id)

        # Save accumulated normalized reward
        self.ep_norm_rewards.append(sum(ep_rewards) / num_agents)

        # Some vehicles may be finished earlier than others, so we mask out the invalid samples
        # And flatten along the agent axis
        valid_samples_mask = ~np.isnan(expert_action_arr)
            
        expert_action_arr = expert_action_arr[valid_samples_mask]
        obs_arr = obs_arr[valid_samples_mask]
        next_obs_arr = next_obs_arr[valid_samples_mask]
        dones_arr = dones_arr[valid_samples_mask].astype(bool)
        
        return obs_arr, expert_action_arr, next_obs_arr, dones_arr

    def _set_discrete_action_space(self):
        """Set the discrete action space."""

        self.action_space = Discrete(self.config.accel_discretization * self.config.steering_discretization)
        self.accel_grid = np.linspace(
            -np.abs(self.config.accel_lower_bound),
            self.config.accel_upper_bound,
            self.config.accel_discretization,
        )
        self.steering_grid = np.linspace(
            -np.abs(self.config.steering_lower_bound),
            self.config.steering_upper_bound,
            self.config.steering_discretization,
        )
        self.joint_idx_to_actions = {}
        self.actions_to_joint_idx = {}
        for i, (accel, steer) in enumerate(product(self.accel_grid, self.steering_grid)):
            self.joint_idx_to_actions[i] = [accel, steer]
            self.actions_to_joint_idx[accel, steer] = [i]

    def _find_closest_index(self, action_grid, action):
        """Find the nearest value in the action grid for a given expert action."""
        indx = np.argmin(np.abs(action_grid - action))
        return action_grid[indx], indx
    
     
if __name__ == "__main__":

    env_config = load_config("env_config")
    env_config.num_files = 1000

    # Create iterator
    waymo_iterator = TrajectoryIterator(
        data_path=env_config.data_path,
        env_config=env_config,
        file_limit=env_config.num_files,
    )   

    # Rollout to get obs-act-obs-done trajectories 
    rollouts = next(iter(
        DataLoader(
            waymo_iterator,
            batch_size=10_000, # Number of samples to generate
            pin_memory=True,
    )))

    obs, acts, next_obs, dones = rollouts

    print('hi')
    
