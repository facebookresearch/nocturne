import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb

from imitation.algorithms import bc
from imitation.data.types import Transitions

from utils.wrappers import LightNocturneEnvWrapper
from utils.config import load_config
from utils.imitation_learning.waymo_iterator import TrajectoryIterator
from utils.evaluation import evaluate_policy


if __name__ == "__main__":

    # Create run
    run = wandb.init(project="eval_il_policy")

    # Configs
    video_config = load_config("video_config")
    env_config = load_config("env_config")
    bc_config = load_config("bc_config")

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
            batch_size=bc_config.total_samples,
            pin_memory=True,
    )))

    # Convert to dataset of imitation "transitions" 
    transitions = Transitions(
        obs=rollouts[0], 
        acts=rollouts[1], 
        infos=torch.zeros_like(rollouts[0]), # Dummy
        next_obs=rollouts[2],
        dones=np.array(rollouts[3]).astype(bool)
    )

    # Define trainer
    rng = np.random.default_rng()
    bc_trainer = bc.BC(
        observation_space=waymo_iterator.observation_space,
        action_space=waymo_iterator.action_space,
        demonstrations=transitions,
        rng=rng,
        device="cpu",
    )

    # Create evaluation env
    env_config.sample_file_method = "no_replacement"

    # Check random behavior
    reward_before_training, _ = evaluate_policy(
        model=bc_trainer.policy, 
        env=LightNocturneEnvWrapper(env_config),
        n_steps_per_episode=env_config.episode_length, 
        n_eval_episodes=env_config.num_files,
        video_config=video_config,
        video_caption="BEFORE training",
        render=True,
    )

    # Train
    bc_trainer.train(
        n_epochs=bc_config.n_epochs,
    )

    # Evaluate
    reward_after_training, _ = evaluate_policy(
        model=bc_trainer.policy, 
        env=LightNocturneEnvWrapper(env_config),
        n_steps_per_episode=env_config.episode_length, 
        n_eval_episodes=env_config.num_files,
        video_config=video_config,
        video_caption=f"AFTER training ({bc_config.n_epochs} epochs)",
        render=True,
    )

    print(f"Reward before training: {reward_before_training:.2f}")
    print(f"Reward after  training: {reward_after_training:.2f}")