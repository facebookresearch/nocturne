import logging
import os
import wandb
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

from nocturne.envs.base_env import BaseEnv
from utils.render import make_video


class CustomMultiAgentCallback(BaseCallback):
    """
    Nocturne-compatible custom callback that derives from ``BaseCallback``.
    """

    def __init__(
        self,
        env_config,
        exp_config,
        video_config=None,
        wandb_run=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.env_config = env_config
        self.exp_config = exp_config
        self.video_config = video_config
        self.iteration = 0
        self.wandb_run = wandb_run
        self.model_base_path = os.path.join(wandb.run.dir, "policies")
        if self.model_base_path is not None:
            os.makedirs(self.model_base_path, exist_ok=True)

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
        self.ep_rewards_avg_norm = sum(rewards.sum(axis=1) / num_agents_per_step) / self.n_episodes

        # Obtain advantages
        advantages = np.nan_to_num(self.locals["rollout_buffer"].advantages, nan=0)
        self.ep_advantage_avg_norm = sum(advantages.sum(axis=1) / num_agents_per_step) / self.n_episodes

        # Obtain the average ratio of agents that collided / achieved goal in the episode
        self.avg_frac_collided = self.locals["env"].num_agents_collided / self.locals["env"].total_agents_in_rollout
        self.avg_frac_goal_achieved = (
            self.locals["env"].num_agents_goal_achieved / self.locals["env"].total_agents_in_rollout
        )

        # Sanity check: log observation min and max
        observations = self.locals["rollout_buffer"].observations
        valid_obs_mask = ~np.isnan(self.locals["rollout_buffer"].observations)

        # Log aggregate performance measures
        self.logger.record("rollout/avg_num_agents_controlled", np.mean(num_agents_per_step))
        self.logger.record("rollout/ep_rew_mean_norm", self.ep_rewards_avg_norm)
        self.logger.record("rollout/ep_len_mean", avg_ep_len)
        self.logger.record("rollout/perc_goal_achieved", (self.avg_frac_goal_achieved) * 100)
        self.logger.record("rollout/perc_collided", (self.avg_frac_collided) * 100)
        self.logger.record("rollout/ep_adv_mean_norm", self.ep_advantage_avg_norm)
        self.logger.record("rollout/global_step", self.num_timesteps)
        self.logger.record("rollout/iter", self.iteration)
        self.logger.record("rollout/obs_min", np.min(observations[valid_obs_mask]))
        self.logger.record("rollout/obs_max", np.max(observations[valid_obs_mask]))

        # Evaluate policy on train and test dataset
        # if self.iteration % self.exp_config.ma_callback.eval_freq == 0:
        #     self._evaluate_policy(policy=self.model, dataset="valid", name="valid_det", det_mode=True)

        # Render
        if self.exp_config.ma_callback.save_video:
            if self.iteration % self.exp_config.ma_callback.video_save_freq == 0:
                logging.info(f"Making video at iter = {self.iteration} | global_step = {self.num_timesteps}")
                make_video(
                    env_config=self.env_config,
                    exp_config=self.exp_config,
                    video_config=self.video_config,
                    filenames=[self.locals["env"].filename],
                    model=self.model,
                    n_steps=self.num_timesteps,
                    deterministic=self.exp_config.ma_callback.video_deterministic,
                )

        # Save model
        if self.exp_config.ma_callback.save_model:
            if self.iteration % self.exp_config.ma_callback.model_save_freq == 0:
                self._save_model()

        # Update probabilities for sampling scenes
        if self.locals["env"].psr_dict is not None:
            self._update_sampling_probs()

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        # Save model to wandb
        if self.model_base_path is not None:
            self._save_model()

        # Render
        if self.exp_config.ma_callback.save_video:
            logging.info(
                f"Making video at last iter = {self.iteration} in deterministic mode | global_step = {self.num_timesteps}"
            )
            make_video(
                env_config=self.env_config,
                exp_config=self.exp_config,
                video_config=self.video_config,
                filenames=[self.locals["env"].filename],
                model=self.model,
                n_steps=self.num_timesteps,
                deterministic=self.exp_config.ma_callback.video_deterministic,
            )

    def _save_model(self) -> None:
        """Save model locally and to wandb."""

        self.path = os.path.join(
            self.model_base_path,
            f"policy_L{self.exp_config.reg_weight}_S{self.env_config.num_files}_I{self.iteration}.zip",
        )
        self.model.save(self.path)
        wandb.save(self.path, base_path=self.model_base_path)
        logging.info(f"Saved policy on step {self.num_timesteps} / iter {self.iteration} at: \n {self.path}")

    def _evaluate_policy(self, policy, dataset="train", name="", det_mode=True, data_folder="data_full", num_files=100):
        """Evaluate policy in a number of scenes."""

        env_config = self.env_config.copy()

        total_coll = 0
        total_goal_achieved = 0
        total_samples = 0
        total_eval_episodes = 0

        # Choose dataset
        if dataset == "train":
            env_config.data_path = f"./{data_folder}/train"
        elif dataset == "valid":
            env_config.data_path = f"./{data_folder}/valid"
            env_config.num_files = num_files

        # Make environment
        env = BaseEnv(env_config)

        # Evaluate policy in a number of scenes
        obs_dict = env.reset()

        agent_ids = [veh_id for veh_id in obs_dict.keys()]
        dead_agent_ids = []
        last_info_dicts = {agent_id: {} for agent_id in agent_ids}

        for _ in range(2000):
            # Get actions
            action_dict = {}
            for agent_id in obs_dict:
                # Get observation
                obs = torch.from_numpy(obs_dict[agent_id]).unsqueeze(dim=0)
                # Get action
                with torch.no_grad():
                    action, _ = policy.predict(obs, deterministic=det_mode)
                # Store action
                action_dict[agent_id] = int(action)

            # Step in the environment
            obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

            for agent_id, is_done in done_dict.items():
                if is_done and agent_id not in dead_agent_ids:
                    dead_agent_ids.append(agent_id)
                    # Store agents' last info dict
                    last_info_dicts[agent_id] = info_dict[agent_id].copy()

            if done_dict["__all__"]:
                # Update stats
                for agent_id in agent_ids:
                    total_coll += last_info_dicts[agent_id]["collided"] * 1
                    total_goal_achieved += last_info_dicts[agent_id]["goal_achieved"] * 1

                total_samples += len(agent_ids)  # num episodes x num agents
                total_eval_episodes += 1

                # Reset
                obs_dict = env.reset()
                agent_ids = [veh_id for veh_id in obs_dict.keys()]
                dead_agent_ids = []
                last_info_dicts = {agent_id: {} for agent_id in agent_ids}

        self.logger.record(f"eval_{name}_/num_eval_scenes", total_eval_episodes)
        self.logger.record(f"eval_{name}_/total_samples", total_samples)
        self.logger.record(f"eval_{name}_/goal_rate", (total_goal_achieved / total_samples) * 100)
        self.logger.record(f"eval_{name}_/coll_rate", (total_coll / total_samples) * 100)

    def _update_sampling_probs(self):
        """Update sampling probabilities for each scene based on the performance of the agent in that scene."""

        # Lower sampling probability for scenes with high goal rate
        for scene in self.locals["env"].psr_dict.keys():
            scene_rew = self.locals["env"].psr_dict[scene]["reward"]
            if scene_rew >= 0.9:
                self.locals["env"].psr_dict[scene]["prob"] = 1e-8  # Assign low probability

        # Sampling probability is inversely proportional to the average reward
        roll_avg_rewards = np.array([item["reward"] for item in self.locals["env"].psr_dict.values()])
        weighted_scenes = np.exp(-roll_avg_rewards)
        probs = np.exp(weighted_scenes - np.max(weighted_scenes)) / np.sum(
            np.exp(weighted_scenes - np.max(weighted_scenes))
        )

        for idx, scene in enumerate(self.locals["env"].psr_dict.keys()):
            self.locals["env"].psr_dict[scene]["prob"] = probs[idx]
