import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from wandb.integration.sb3 import WandbCallback
from utils.render_utils import save_nocturne_video


class CustomMultiAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(
        self,
        env_config,
        exp_config,
        video_config=None,
        save_video_callbacks=None,
        training_end_callbacks=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.env_config = env_config
        self.exp_config = exp_config
        self.video_config = video_config
        self.save_video_callbacks = (
            [] if save_video_callbacks is None else save_video_callbacks
        )
        self.training_end_callbacks = (
            [] if training_end_callbacks is None else training_end_callbacks
        )
        self.iteration = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.
        """
        pass

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        # # Compute the number of episodes completed during this rollout
        self.n_episodes = self.locals["env"].n_episodes

        # Every rollout end marks an iteration
        self.iteration += 1

        # Compute average episode length across all agents
        avg_ep_len = np.mean(self.locals["env"].episode_lengths)

        # Get rewards, filter out NaNs
        rewards = np.nan_to_num(self.locals["rollout_buffer"].rewards, nan=0)

        # Get number of agents in scene 
        #TODO: Only works when every scene has the same number of agents
        num_agents_in_scene = len(self.locals["env"].agent_ids)

        # Obtain the average reward obtained per episode across agents
        avg_rewards = rewards.sum() / self.n_episodes

        # Compute average reward obtained per agent, per episode
        if self.exp_config.custom_callback.log_indiv_rewards:
            indiv_rewards_sum = rewards.sum(axis=0)[:num_agents_in_scene]
            for agent_idx in range(num_agents_in_scene):
                self.logger.record(
                    f"rollout/ep_rew_mean_agent_{agent_idx}",
                    indiv_rewards_sum[agent_idx] / self.n_episodes
                )

        # Get batch size
        batch_size = (~np.isnan(self.locals["rollout_buffer"].rewards)).sum()

        # Get fraction of agents collided & goal achieved per episode
        frac_collided = (self.locals["env"].collided / num_agents_in_scene) / self.n_episodes
        frac_goal_achieved = (self.locals["env"].goal_achieved / num_agents_in_scene) / self.n_episodes

        # Log
        self.logger.record("rollout/ep_rew_mean", avg_rewards)
        self.logger.record("rollout/ep_rew_std", rewards.std())
        self.logger.record("rollout/ep_len_mean", avg_ep_len)
        self.logger.record("rollout/perc_goal_achieved", frac_goal_achieved)
        self.logger.record("rollout/perc_collided", frac_collided)
        # The global step is defined as the number of individual steps in the env
        self.logger.record("global_step", self.num_timesteps)
        self.logger.record("iteration", self.iteration)
        self.logger.record("num_frames_in_rollout", batch_size)

        # Reset number of episodes per rollout
        if self.exp_config.custom_callback.save_video:
            if self.iteration % self.exp_config.custom_callback.video_save_freq == 0:
                logging.info(
                    f"Make video at iter = {self.iteration} | global_step = {self.num_timesteps}"
                )
                save_nocturne_video(
                    env_config=self.env_config,
                    exp_config=self.exp_config,
                    video_config=self.video_config,
                    model=self.model,
                    n_steps=self.num_timesteps,
                    deterministic=self.exp_config.custom_callback.video_deterministic,
                )

        #NOTE: RESET COLLIDED AND GOAL ACHIEVED (tmp solution, this doesn't belong in the callback)
        self.locals["env"].collided = 0
        self.locals["env"].goal_achieved = 0
        self.locals["env"].n_episodes = 0
        self.locals["env"].episode_lengths = []

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        super()._on_training_end()
        for training_end_callback in self.training_end_callbacks:
            training_end_callback(self.model)