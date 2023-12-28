"""Cast a multi-agent env as vec env to use SB3's PPO."""
import logging
from datetime import datetime
from contextlib import nullcontext
from box import Box
import torch
import numpy as np
import wandb
from stable_baselines3.common.policies import ActorCriticPolicy

# Import networks

from networks.mlp_late_fusion import LateFusionMLP, LateFusionMLPPolicy

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from utils.render import make_video

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.reg_ppo import RegularizedPPO
from utils.string_utils import datetime_to_str
from utils.random_utils import init_seed

logging.basicConfig(level=logging.INFO)

def train(env_config, exp_config, video_config):
    """Train RL agent using PPO."""

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
    RUN_ID = f"{datetime_}" if exp_config.track_wandb else None

    # Add scene to config
    exp_config.scene = env.filename

    with wandb.init(
        project=exp_config.project,
        name=RUN_ID,
        config={**exp_config, **env_config},
        id=RUN_ID,
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
            wandb_run=run if RUN_ID is not None else None,
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
            reg_weight=exp_config.reg_weight, # Regularization weight; lambda
            env=env,
            n_steps=exp_config.ppo.n_steps,
            policy=LateFusionMLPPolicy, 
            ent_coef=exp_config.ppo.ent_coef,
            vf_coef=exp_config.ppo.vf_coef,
            seed=exp_config.seed,  # Seed for the pseudo random generators
            verbose=exp_config.verbose,
            tensorboard_log=f"runs/{RUN_ID}" if RUN_ID is not None else None,
            device=exp_config.ppo.device,
            env_config=env_config,
            mlp_class=LateFusionMLP,
            mlp_config=Box(
                {
                    "arch_ego_state": [8],
                    "arch_road_objects": [64],
                    "arch_road_graph": [126, 64],
                    "act_func": "tanh", 
                    "last_layer_dim_pi": 64,
                    "last_layer_dim_vf": 64,
                }
            )
        )

        # Log number of trainable parameters
        policy_params = filter(lambda p: p.requires_grad, model.policy.parameters())
        params = sum([np.prod(p.size()) for p in policy_params])
        exp_config.n_policy_params = params
        logging.info(f'Policy | trainable params: {params:,} \n')

        # Architecture
        logging.info(f'Policy | arch: \n {model.policy}')

        # Learn
        model.learn(
            **exp_config.learn,
            callback=custom_callback,
        )

if __name__ == "__main__":

    # Load environment and experiment configurations
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    video_config = load_config("video_config")
    
    lambdas = [0.01]
    for lam in lambdas:
        # Set regularization weight
        exp_config.reg_weight = lam

    # Train
    train(
        env_config, 
        exp_config, 
        video_config,
    )