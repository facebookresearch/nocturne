"""Train HR-PPO agent with CLI arguments."""
import logging
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import torch
import typer
from box import Box
from stable_baselines3.common.policies import ActorCriticPolicy

import wandb

# Import networks
from networks.mlp_late_fusion import LateFusionMLP, LateFusionMLPPolicy

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from utils.random_utils import init_seed
from utils.render import make_video

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.reg_ppo import RegularizedPPO
from utils.string_utils import datetime_to_str

logging.basicConfig(level=logging.INFO)

# Default settings
env_config = load_config("env_config")
exp_config = load_config("exp_config")
video_config = load_config("video_config")

LAYERS_DICT = {
    "tiny": [64],
    "small": [128, 64],
    "medium": [256, 128],
    "large": [256, 128, 64],
}


def run_hr_ppo(
    sweep_name: str = "hr_ppo",
    steer_disc: int = 5,
    accel_disc: int = 5,
    ent_coef: float = 0.0,
    vf_coef: float = 0.5,
    seed: int = 42,
    arch_road_objects: str = "small",
    arch_road_graph: str = "small",
    arch_shared_net: str = "small",
    activation_fn: str = "tanh",
    dropout: float = 0.0, 
    total_timesteps: int = 1_000_000,
    num_files: int = 10,
    reg_weight: float = 0.0,
) -> None:
    """Train RL agent using PPO with CLI arguments."""
    # ==== Overwrite default settings ==== #
    # Environment
    env_config.steer_disc = steer_disc
    env_config.accel_disc = accel_disc
    env_config.num_files = num_files
    # Experiment
    exp_config.seed = seed
    exp_config.ent_coef = ent_coef
    exp_config.vf_coef = vf_coef
    exp_config.learn.total_timesteps = total_timesteps
    exp_config.reg_weight = reg_weight
    # Model architecture 
    exp_config.model_config.arch_ro = arch_road_objects
    exp_config.model_config.arch_rg = arch_road_graph
    exp_config.model_config.arch_shared = arch_shared_net
    exp_config.model_config.act_func = activation_fn

    # Define model architecture
    model_config = Box(
        {
            "arch_road_objects": LAYERS_DICT[arch_road_objects],
            "arch_road_graph": LAYERS_DICT[arch_road_graph],
            "arch_shared_net": LAYERS_DICT[arch_shared_net],
            "act_func": activation_fn,
            "dropout": dropout,
        }
    )
    # ==== Overwrite default settings ==== #

    # Ensure reproducability
    init_seed(env_config, exp_config, exp_config.seed)

    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config,
        num_envs=env_config.max_num_vehicles,
        train_on_single_scene=exp_config.train_on_single_scene,
    )

    # Set up run
    datetime_ = datetime_to_str(dt=datetime.now())
    run_id = f"{datetime_}" if exp_config.track_wandb else None

    # Add scene to config
    exp_config.scene = env.filename

    with wandb.init(
        project=exp_config.project,
        name=run_id,
        group=sweep_name,
        config={**exp_config, **env_config, **model_config},
        id=run_id,
        **exp_config.wandb,
    ) if exp_config.track_wandb else nullcontext() as run:
        # Set device
        exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")
        logging.info(f"Learning in {len(env.env.files)} scene(s): {env.env.files} | using {exp_config.ppo.device}")
        logging.info(f"--- obs_space: {env.observation_space.shape[0]} ---")
        logging.info(f"Action_space\n: {env.env.idx_to_actions}")

        # Initialize custom callback
        custom_callback = CustomMultiAgentCallback(
            env_config=env_config,
            exp_config=exp_config,
            video_config=video_config,
            wandb_run=run if run_id is not None else None,
        )

        # Make scene init video to check expert actions
        if exp_config.track_wandb:
            for model in exp_config.wandb_init_videos:
                make_video(
                    env_config=env_config,
                    exp_config=exp_config,
                    video_config=video_config,
                    filenames=[env.filename],
                    model=model,
                    n_steps=None,
                )

        # Load human reference policy
        saved_variables = torch.load(exp_config.human_policy_path, map_location=exp_config.ppo.device)
        human_policy = ActorCriticPolicy(**saved_variables["data"])
        human_policy.load_state_dict(saved_variables["state_dict"])
        human_policy.to(exp_config.ppo.device)

        # Set up PPO
        model = RegularizedPPO(
            reg_policy=human_policy,
            reg_weight=exp_config.reg_weight,  # Regularization weight; lambda
            env=env,
            n_steps=exp_config.ppo.n_steps,
            policy=LateFusionMLPPolicy,
            ent_coef=exp_config.ppo.ent_coef,
            vf_coef=exp_config.ppo.vf_coef,
            seed=exp_config.seed,  # Seed for the pseudo random generators
            verbose=exp_config.verbose,
            tensorboard_log=f"runs/{run_id}" if run_id is not None else None,
            device=exp_config.ppo.device,
            env_config=env_config,
            mlp_class=LateFusionMLP,
            mlp_config=model_config,
        )

        # Log number of trainable parameters
        policy_params = filter(lambda p: p.requires_grad, model.policy.parameters())
        params = sum(np.prod(p.size()) for p in policy_params)
        exp_config.n_policy_params = params
        logging.info(f"Policy | trainable params: {params:,} \n")

        # Architecture
        logging.info(f"Policy | arch: \n {model.policy}")

        # Learn
        model.learn(
            **exp_config.learn,
            callback=custom_callback,
        )


if __name__ == "__main__":
    # Run
    typer.run(run_hr_ppo)
