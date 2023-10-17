import logging

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

import wandb
from utils.render import save_nocturne_video


class CustomMultiAgentCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.
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
        self.save_video_callbacks = [] if save_video_callbacks is None else save_video_callbacks
        self.training_end_callbacks = [] if training_end_callbacks is None else training_end_callbacks
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

        # Every rollout end (+ optim step) marks an iteration
        self.iteration += 1

        # Compute average episode length across all agents
        avg_ep_len = np.mean(self.locals["env"].episode_lengths)

        # Get rewards, filter out NaNs
        rewards = np.nan_to_num(self.locals["rollout_buffer"].rewards, nan=0)

        # Average normalized by the number of agents in the scene
        num_agents_per_step = np.array(self.locals["env"].agents_in_scene)
        ep_rewards_avg_norm = sum(rewards.sum(axis=1) / num_agents_per_step) / self.n_episodes

        # Obtain the sum of reward per episode (accross all agents)
        sum_rewards = rewards.sum() / self.n_episodes

        # Get batch size
        batch_size = (~np.isnan(self.locals["rollout_buffer"].rewards)).sum()

        # Obtain the average ratio of agents that collided / achieved goal in the episode
        avg_frac_collided = np.mean(self.locals["env"].frac_collided)
        avg_frac_goal_achieved = np.mean(self.locals["env"].frac_goal_achieved)

        # Log
        if self.exp_config.track_wandb:
            agent_bins = np.arange(0, self.locals["env"].num_envs + 1, 1)
            hist = np.histogram(num_agents_per_step, bins=agent_bins)
            wandb.log({"rollout/dist_agents_in_scene": wandb.Histogram(np_histogram=hist)})
        self.logger.record("rollout/ep_rew_mean_norm", ep_rewards_avg_norm)
        self.logger.record("rollout/ep_rew_sum", sum_rewards)
        self.logger.record("rollout/ep_rew_std", rewards.std())
        self.logger.record("rollout/ep_len_mean", avg_ep_len)
        self.logger.record("rollout/perc_goal_achieved", avg_frac_goal_achieved)
        self.logger.record("rollout/perc_collided", avg_frac_collided)
        self.logger.record("global_step", self.num_timesteps)
        self.logger.record("iteration", self.iteration)
        self.logger.record("num_frames_in_rollout", batch_size)

        # Make a video with a random scene
        if self.exp_config.ma_callback.save_video:
            if self.iteration % self.exp_config.ma_callback.video_save_freq == 0:
                logging.info(f"Making video at iter = {self.iteration} | global_step = {self.num_timesteps}")
                save_nocturne_video(
                    env_config=self.env_config,
                    exp_config=self.exp_config,
                    video_config=self.video_config,
                    model=self.model,
                    n_steps=self.num_timesteps,
                    deterministic=self.exp_config.ma_callback.video_deterministic,
                )

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        super()._on_training_end()
        for training_end_callback in self.training_end_callbacks:
            training_end_callback(self.model)
