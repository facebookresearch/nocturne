"""Module containing regularized PPO algorithm."""

import logging

import numpy as np
import torch
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

# Import masked buffer class
from algorithms.masked_buffer import MaskedRolloutBuffer

logging.getLogger(__name__)


class CustomPPO(PPO):
    """Adapted Proximal Policy Optimization algorithm (PPO) that is compatible with multi-agent environments.
    """

    def _setup_model(self) -> None:
        super()._setup_model()

        # Change buffer to our own masked version
        buffer_cls = MaskedRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MaskedRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Adapted collect_rollouts function."""

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if (
                self.use_sde
                and self.sde_sample_freq > 0
                and n_steps % self.sde_sample_freq == 0
            ):
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # EDIT_1: Check if there is at least one dead agent, if so, mask out observations
                if env.dead_agent_ids:
                    # Create dummy actions, values and log_probs (NaN)
                    actions = torch.full(fill_value=np.nan, size=(self.n_envs,))
                    log_probs = torch.full(fill_value=np.nan, size=(self.n_envs,))
                    values = torch.full(
                        fill_value=np.nan, size=(self.n_envs,)
                    ).unsqueeze(dim=1)
                
                    for idx, agent_id in enumerate(env.agent_ids):
                        if agent_id not in env.dead_agent_ids:
                            # Sample actions from policy if agent is alive
                            obs_tensor_agent_id = obs_tensor[idx, :].unsqueeze(dim=0) 
                            actions[idx], values[idx], log_probs[idx] = self.policy(obs_tensor_agent_id)
                else:
                    # Sample actions from policy
                    actions, values, log_probs = self.policy(obs_tensor)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(
                        actions, self.action_space.low, self.action_space.high
                    )

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # EDIT_2: Increment the global step by the number of valid samples in rollout step
            samples_in_timestep = env.num_envs - np.isnan(dones).sum()
            self.num_timesteps += samples_in_timestep

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        return True