import numpy as np
from torch.utils.data import DataLoader
import wandb
import torch
from pathlib import Path
from datetime import datetime

from imitation.algorithms import bc
from imitation.data.types import Transitions
from imitation.policies.serialize import save_stable_model

from utils.wrappers import LightNocturneEnvWrapper
from utils.config import load_config
from utils.imitation_learning.waymo_iterator import TrajectoryIterator
from utils.evaluation import evaluate_policy

from utils.string_utils import date_to_str

if __name__ == "__main__":

    # Create run
    run = wandb.init(
        project="eval_il_policy",
        sync_tensorboard=True,
    )

    # Configs
    video_config = load_config("video_config")
    env_config = load_config("env_config")
    bc_config = load_config("bc_config")

    # Device TODO: Add support for CUDA
    device = "cpu" 

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
        obs=rollouts[0].to(device),
        acts=rollouts[1].to(device), 
        infos=np.zeros_like(rollouts[0]), # Dummy
        next_obs=rollouts[2],
        dones=np.array(rollouts[3]).astype(bool),
    )

    # Define trainer
    rng = np.random.default_rng()
    bc_trainer = bc.BC(
        observation_space=waymo_iterator.observation_space,
        action_space=waymo_iterator.action_space,
        demonstrations=transitions,
        rng=rng,
        device=device,
    )

    # Create evaluation env
    env = LightNocturneEnvWrapper(env_config)
    eval_files = env.files

    # Check random behavior
    reward_before_training, _ = evaluate_policy(
        model=bc_trainer.policy, 
        env=LightNocturneEnvWrapper(env_config),
        n_steps_per_episode=env_config.episode_length, 
        n_eval_episodes=1,
        eval_files=eval_files,
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
        n_eval_episodes=1,
        eval_files=eval_files,
        video_config=video_config,
        video_caption=f"AFTER training ({bc_config.n_epochs} epochs)",
        render=True,
    )

    print(f"Reward before training: {reward_before_training:.2f}")
    print(f"Reward after  training: {reward_after_training:.2f}")

    # Save model
    date_ = date_to_str(datetime.now())

    if bc_config.save_model:
        bc_trainer.policy.save(path=f"{bc_config.save_model_path}{bc_config.model_name}_{date_}.pt")