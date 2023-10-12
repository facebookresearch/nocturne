"""Vectorized environment wrapper for multi-agent environments."""
import logging
import time
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.util import (
    copy_obs_dict,
    dict_to_obs,
    obs_space_info,
)
from nocturne.envs.base_env import BaseEnv
from utils.config import load_config

logging.basicConfig(level=logging.INFO)


class MultiAgentAsVecEnv(VecEnv):
    """A wrapper that casts multi-agent environments as vectorized environments.

    Args:
    -----
        VecEnv (SB3 VecEnv): SB3 VecEnv base class.
    """

    def __init__(self, config, num_envs):
        # Create Nocturne env
        self.env = BaseEnv(config)

        # Make action and observation spaces compatible with SB3 (requires gymnasium)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.env.observation_space.shape, np.float32)
        self.num_envs = num_envs  # The maximum number of agents allowed in the environment
        self.keys, shapes, dtypes = obs_space_info(self.env.observation_space)

        # Storage
        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k])) for k in self.keys]
        )
        self.buf_dones = np.full(fill_value=np.nan, shape=(self.num_envs,))
        self.buf_rews = np.full(fill_value=np.nan, shape=(self.num_envs,))
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.n_episodes = 0
        self.episode_lengths = []
        self.rewards = []  # Log reward per step
        self.dead_agent_ids = []  # Log dead agents per step
        self.frac_collided = []  # Log fraction of agents that collided
        self.frac_goal_achieved = []  # Log fraction of agents that achieved their goal
        self.agents_in_scene = []

    def _reset_seeds(self) -> None:
        """Reset all environments' seeds."""
        self._seeds = None

    def reset(self, seed=None):
        """Reset environment and return initial observations."""
        # Reset Nocturne env
        obs_dict = self.env.reset()

        # Reset storage
        self.agent_ids = []
        self.rewards = []
        self.dead_agent_ids = []
        self.ep_collisions = 0
        self.ep_goal_achived = 0

        obs_all = np.full(fill_value=np.nan, shape=(self.num_envs, self.env.observation_space.shape[0]))
        for idx, agent_id in enumerate(obs_dict.keys()):
            self.agent_ids.append(agent_id)
            obs_all[idx, :] = obs_dict[agent_id]

        # Save obs in buffer
        self._save_obs(obs_all)

        logging.debug(f"RESET - agent ids: {self.agent_ids}")

        # Make dict for storing the last info set for each agent
        self.last_info_dicts = {agent_id: {} for agent_id in self.agent_ids}

        return self._obs_from_buf()

    def step(self, actions) -> VecEnvStepReturn:
        """Convert action vector to dict and call env.step()."""

        agent_actions = {
            agent_id: actions[idx] for idx, agent_id in enumerate(self.agent_ids) if agent_id not in self.dead_agent_ids
        }

        # Take a step to obtain dicts
        next_obses_dict, rew_dict, done_dict, info_dict = self.env.step(agent_actions)

        # Update dead agents based on most recent done_dict
        for agent_id, is_done in done_dict.items():
            if is_done and agent_id not in self.dead_agent_ids:
                self.dead_agent_ids.append(agent_id)
                # Store agents' last info dict
                self.last_info_dicts[agent_id] = info_dict[agent_id].copy()

        # Convert dicts to arrays
        obs_all = np.full(
            fill_value=np.nan,
            shape=(self.num_envs, self.env.observation_space.shape[0]),
        )
        rew_all = np.full(fill_value=np.nan, shape=(self.num_envs))
        done_all = np.full(fill_value=np.nan, shape=(self.num_envs))
        info_all = []

        for idx, key in enumerate(self.agent_ids):
            # Store data if available; otherwise leave as NaN
            if key in next_obses_dict:
                obs_all[idx, :] = next_obses_dict[key]
                rew_all[idx] = rew_dict[key]
                done_all[idx] = done_dict[key] * 1  # Will be 0 or 1 if valid, NaN otherwise

        # OVERRIDE old buffer vals with with new ones (for all envs)
        for env_idx in range(self.num_envs):
            info_all.append(info_dict[key])
            self.buf_rews[env_idx] = rew_all[env_idx]
            self.buf_dones[env_idx] = done_all[env_idx]
            self.buf_infos[env_idx] = info_all[env_idx]

        # Save step reward obtained across all agents
        self.rewards.append(sum(rew_dict.values()))
        self.agents_in_scene.append(len(self.agent_ids))

        # O(t) = O(t+1)
        self._save_obs(obs_all)

        # Reset episode if ALL agents are done
        if done_dict["__all__"]:
            for agent_id in self.agent_ids:
                # Store total number of collisions and goal achievements across rollout
                self.ep_collisions += self.last_info_dicts[agent_id]["collided"] * 1
                self.ep_goal_achived += self.last_info_dicts[agent_id]["goal_achieved"] * 1

            # Store the fraction of agents that collided in episode
            self.frac_collided.append(self.ep_collisions / len(self.agent_ids))
            self.frac_goal_achieved.append(self.ep_goal_achived / len(self.agent_ids))

            # Save final observation where user can get it, then reset
            for env_idx in range(len(self.agent_ids)):
                self.buf_infos[env_idx]["terminal_observation"] = obs_all[env_idx]

            # Log episode stats
            ep_len = self.step_num
            self.n_episodes += 1
            self.episode_lengths.append(ep_len)

            # Reset
            obs_all = self.reset()

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def close(self) -> None:
        """Close the environment."""
        self.env.close()

    @property
    def step_num(self) -> List[int]:
        """The episodic timestep."""
        return self.env.step_num

    def seed(self, seed=None):
        """Set the random seeds for all environments."""
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def _save_obs(self, obs: VecEnvObs) -> None:
        """Save observations into buffer."""
        for key in self.keys:
            if key is None:
                self.buf_obs[key] = obs
            else:
                self.buf_obs[key] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
        """Get observation from buffer."""
        return dict_to_obs(self.observation_space, copy_obs_dict(self.buf_obs))

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def env_is_wrapped(self, wrapper_class, indices=None):
        raise NotImplementedError()

    def step_async(self, actions: np.ndarray) -> None:
        raise NotImplementedError()

    def step_wait(self) -> VecEnvStepReturn:
        raise NotImplementedError()


if __name__ == "__main__":
    MAX_AGENTS = 3
    NUM_STEPS = 400

    # Load environment variables and config
    env_config = load_config("env_config")

    # Set the number of max vehicles
    env_config.max_num_vehicles = MAX_AGENTS

    # Make environment
    env = MultiAgentAsVecEnv(config=env_config, num_envs=MAX_AGENTS)

    obs = env.reset()
    for global_step in range(NUM_STEPS):
        # Take random action(s) -- you'd obtain this from a policy
        actions = np.array([env.env.action_space.sample() for _ in range(MAX_AGENTS)])

        # Step
        obs, rew, done, info = env.step(actions)

        # Log
        logging.info(f"step_num: {env.step_num} (global = {global_step}) | done: {done} | rew: {rew}")

        time.sleep(0.2)