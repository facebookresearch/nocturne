import logging
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

import gymnasium as gym
import numpy as np
import torch
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
    """
    NOTE: CURRENTLY SUPPORTS ONLY A SINGLE AGENT
    Wrapper that treats an environment with multiple agents as vectorized environment.
    """

    def __init__(self, config, num_envs):
        # Create Nocturne env
        self.env = BaseEnv(config)

        # Make action and observation spaces compatible with SB3 (requires gymnasium)
        self.action_space = gym.spaces.Discrete(self.env.action_space.n)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, self.env.observation_space.shape, np.float32
        )
        self.num_envs = num_envs  # The number of agents
        self.keys, shapes, dtypes = obs_space_info(self.env.observation_space)

        # Storage
        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((self.num_envs, *tuple(shapes[k])), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((self.num_envs,))
        self.buf_rews = np.zeros((self.num_envs,))
        self.buf_infos: List[Dict[str, Any]] = [{} for _ in range(self.num_envs)]
        self.metadata = self.env.metadata
        self.n_episodes = 0
        self.rewards = []         # Log reward per step 
        self.dead_agent_ids = []  # Log dead agents per step
        self.collided = [] # Log if agent collided at episode end
        self.goal_achieved = [] # Log if agent achieved goal at episode end

    def _reset_seeds(self) -> None:
        self._seeds = None

    def reset(self, seed=None):
        # Reset Nocturne env
        obs_dict = self.env.reset()
        self.agent_ids = []
        self.rewards = []
        self.dead_agent_ids = []
        obs_all = np.zeros((self.num_envs, self.env.observation_space.shape[0]))
        for idx, agent_id in enumerate(obs_dict.keys()):
            self.agent_ids.append(agent_id)
            obs_all[idx, :] = obs_dict[agent_id]

        # Save obs in buffer
        self._save_obs(obs_all)
        # Seeds are only used once
        # self._reset_seeds()
        self.agent_mapping = {
            agent_id: agent_idx for agent_idx, agent_id in enumerate(self.agent_ids)
        }
        return self._obs_from_buf()

    def close(self) -> None:
        self.env.close()

    def step(self, actions) -> VecEnvStepReturn:
        """Convert action vector to dict and call env.step()."""

        agent_actions = {
            agent_id: actions[idx]
            for idx, agent_id in enumerate(self.agent_ids)
            if agent_id not in self.dead_agent_ids
        }

        # Take a step in the env to obtain obs dict
        next_obses_dict, rew_dict, done_dict, info_dict = self.env.step(agent_actions)

        # Update dead agents
        for agent_id, is_done in done_dict.items():
            if is_done and agent_id not in self.dead_agent_ids:
                self.dead_agent_ids.append(agent_id)

        # Storage
        obs_all = np.full(
            fill_value=np.nan,
            shape=(self.num_envs, self.env.observation_space.shape[0]),
        )
        rew_all = np.full(fill_value=np.nan, shape=(self.num_envs))
        done_all = np.full(fill_value=np.nan, shape=(self.num_envs))
        info_all = []

        for idx, key in enumerate(self.agent_ids):
            info_all.append(info_dict[key])
            # If agent is still active store data
            if key in next_obses_dict:
                obs_all[idx, :] = next_obses_dict[key]
                rew_all[idx] = rew_dict[key]
                done_all[idx] = done_dict[key] * 1

        # Store
        self.rewards.append(sum(rew_dict.values()))
        for env_idx in range(self.num_envs):
            self.buf_rews[env_idx] = rew_all[env_idx]
            self.buf_dones[env_idx] = done_all[env_idx]
            self.buf_infos[env_idx] = info_all[env_idx]

        # O(t) = O(t+1)
        self._save_obs(obs_all)

        # If all agents are done or we reached the end of the episode,
        # store last observation and reset
        # NOTE: TMP EDIT TO TEST MASKING - RESET ONLY AT EPISODE END
        if done_dict["__all__"]:
        # if self.step_num == 80:
            for agent_id in self.agent_ids:
                self.collided.append(info_dict[agent_id]["collided"])
                self.goal_achieved.append(info_dict[agent_id]["goal_achieved"])

            # Save final observation where user can get it, then reset
            for env_idx in range(self.num_envs):
                self.buf_infos[env_idx]["terminal_observation"] = obs_all[env_idx]
            ep_rew = sum(self.rewards)
            ep_len = len(self.rewards)
            self.n_episodes += 1

            logging.info(
                f"Episode done at step: {self.step_num} | ep_rew = {ep_rew:.2f} | ep_len = {ep_len} | ids: {self.dead_agent_ids}"
            )

            # Reset environment
            obs_all = self.reset()

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    @property
    def step_num(self) -> List[int]:
        """The episodic timestep."""
        return self.env.step_num

    def seed(self, seed=None):
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def _save_obs(self, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key] = obs
            else:
                self.buf_obs[key] = obs[key]  # type: ignore[call-overload]

    def _obs_from_buf(self) -> VecEnvObs:
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

    def get_images(self):
        raise NotImplementedError()


if __name__ == "__main__":
    # NOTE: Currently only supports settings where MAX_AGENTS == number of agents in the scene
    MAX_AGENTS = 1
    NUM_STEPS = 1000

    # Load environment variables and config
    env_config = load_config("env_config")

    # Ensure we only have a single agent
    env_config.max_num_vehicles = MAX_AGENTS

    # Make environment
    env = MultiAgentAsVecEnv(config=env_config, num_envs=MAX_AGENTS)

    obs = env.reset()
    for _ in range(NUM_STEPS):
        # Take random action(s) -- you'd obtain this from a policy
        actions = np.array([env.env.action_space.sample() for _ in range(MAX_AGENTS)])

        # Step
        obs, rew, done, info = env.step(actions)

        # Log
        logging.debug(
            f"step_num: {env.step_num} | done: {done} | cum_rew: {sum(env.rewards):.3f}"
        )