import logging
import torch
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import os

import wandb
from utils.render import make_video


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
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.env_config = env_config
        self.exp_config = exp_config
        self.video_config = video_config
        self.save_video_callbacks = [] if save_video_callbacks is None else save_video_callbacks
        self.training_end_callbacks = [] if training_end_callbacks is None else training_end_callbacks
        self.iteration = 0
        self.wandb_run = wandb_run
        self.new_artifact = True

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
        # Compute the number of episodes completed during this rollout
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

        # Obtain advantages
        advantages = np.nan_to_num(self.locals["rollout_buffer"].advantages, nan=0)
        self.ep_advantage_avg_norm = sum(advantages.sum(axis=1) / num_agents_per_step) / self.n_episodes

        # Get batch size
        batch_size = (~np.isnan(self.locals["rollout_buffer"].rewards)).sum()

        # Obtain the average ratio of agents that collided / achieved goal in the episode
        self.avg_frac_collided = np.mean(self.locals["env"].frac_collided)
        self.avg_frac_goal_achieved = np.mean(self.locals["env"].frac_goal_achieved)

        # Log
        if self.exp_config.track_wandb:
            agent_bins = np.arange(0, self.locals["env"].num_envs + 1, 1)
            hist = np.histogram(num_agents_per_step, bins=agent_bins)
            wandb.log({"rollout/dist_agents_in_scene": wandb.Histogram(np_histogram=hist)})
        
        # Log all metrics on the level of individual agents
        if self.exp_config.ma_callback.log_indiv_metrics and self.env_config.num_files < 2:
            indiv_rewards = ((rewards.sum(axis=0)) / self.n_episodes)
            indiv_advantages = ((advantages.sum(axis=0)) / self.n_episodes)
            for agent_idx in range(len(indiv_rewards)):
                self.logger.record(f"rollout/ep_rew_agent_{agent_idx}", indiv_rewards[agent_idx])
                self.logger.record(f"rollout/ep_adv_agent_{agent_idx}", indiv_advantages[agent_idx])
    
        # Log aggregate performance measures 
        self.logger.record("rollout/avg_num_agents_controlled", np.mean(num_agents_per_step))
        self.logger.record("rollout/ep_rew_mean_norm", ep_rewards_avg_norm)
        self.logger.record("rollout/ep_rew_sum", sum_rewards)
        self.logger.record("rollout/ep_len_mean", avg_ep_len)
        self.logger.record("rollout/perc_goal_achieved", self.avg_frac_goal_achieved)
        self.logger.record("rollout/perc_collided", self.avg_frac_collided)
        self.logger.record("rollout/ep_adv_mean_norm", self.ep_advantage_avg_norm)
        self.logger.record("global_step", self.num_timesteps)
        self.logger.record("iteration", self.iteration)
        self.logger.record("num_frames_in_rollout", batch_size)

        # Make a video with a random scene
        if self.exp_config.ma_callback.save_video:
            if (self.iteration - 1) % self.exp_config.ma_callback.video_save_freq == 0:
                logging.info(f"Making video at iter = {self.iteration} | global_step = {self.num_timesteps}")
                make_video(
                    env_config=self.env_config,
                    exp_config=self.exp_config,
                    video_config=self.video_config,
                    model=self.model,
                    n_steps=self.num_timesteps,
                    deterministic=self.exp_config.ma_callback.video_deterministic,
                )

        # Save model
        if self.exp_config.ma_callback.save_model:
            if self.iteration % self.exp_config.ma_callback.model_save_freq == 0:
                self.save_model()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        if self.model_path is not None:
            self.save_model()
        logging.info(f"-- Saved model artifact at iter {self.iteration} --")

    def save_model(self) -> None:
        """Save model to wandb."""
        self.model_name = f"ppo_policy_net_{self.num_timesteps}"
        self.model_path = os.path.join(wandb.run.dir, f"{self.model_name}.pt")
    
        # Create model artifact
        model_artifact = wandb.Artifact(
            name=self.model_name,
            type="model",
            metadata={**self.env_config, **self.exp_config},
        )

        # Save torch model
        torch.save(
            obj={
                "iter": self.iteration,
                "model_state_dict": self.locals["self"].policy.state_dict(),
                "obs_space_dim": self.locals["env"].observation_space.shape[0],
                "act_space_dim": self.locals["env"].action_space.n,
                "norm_reward": self.ep_advantage_avg_norm,
                "collision_rate": self.avg_frac_collided,
                "goal_rate": self.avg_frac_collided,
            },
            f=self.model_path,
        )

        # Save model artifact
        model_artifact.add_file(local_path=self.model_path)
        wandb.save(self.model_path, base_path=wandb.run.dir)
        self.wandb_run.log_artifact(model_artifact)
        logging.info(f"Saving model checkpoint to {self.model_path} | Global_step: {self.num_timesteps}")