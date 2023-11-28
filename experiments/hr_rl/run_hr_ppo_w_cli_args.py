"""Cast a multi-agent env as vec env to use SB3's PPO."""
import logging
from datetime import datetime
from typing import Callable
import typer
import torch
import wandb
from gymnasium import spaces
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy

# Import networks
from networks.permeq import PermEqNetwork

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

# Default settings
env_config = load_config("env_config")
exp_config = load_config("exp_config")
video_config = load_config("video_config")

POLICY_SIZE_DICT = {
    "small": [126, 64], 
    "medium": [256, 128, 64], 
    "large": [512, 128, 64],
}
POLICY_TYPE_DICT = {
    "mlp": "MlpPolicy",
    "sep_mlp": "SepMlpPolicy",
}

def run_hr_ppo(
    sweep_name: str="hr_ppo",
    steer_disc: int=5, 
    accel_disc: int=5, 
    ent_coef: float=0.,
    vf_coef: float=0.5,
    seed: int=42,
    policy_arch: str="sep_mlp",
    policy_size: str="small",
    activation_fn: str="tanh",
    total_timesteps: int=1_000_000,
    num_files: int=10,
    single_scene: int=0,
    reg_weight: float=0.0,
) -> None:
    """Train RL agent using PPO with CLI arguments."""

    # ==== Update run params ==== #
    # Environment
    env_config.steer_disc = steer_disc
    env_config.accel_disc = accel_disc
    env_config.num_files = num_files
    # Experiment
    exp_config.ent_coef = ent_coef
    exp_config.vf_coef = vf_coef
    exp_config.learn.total_timesteps = total_timesteps
    exp_config.train_on_single_scene = single_scene
    exp_config.policy_arch = policy_arch
    exp_config.policy_size = policy_size
    exp_config.activation_func = activation_fn
    exp_config.reg_weight = reg_weight
    
    # Build policy
    class PermEqActorCriticPolicy(ActorCriticPolicy):
        def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
        ):
            # Disable orthogonal initialization
            kwargs["ortho_init"] = False
            super().__init__(
                observation_space,
                action_space,
                lr_schedule,
                # Pass remaining arguments to base class
                *args,
                **kwargs,
            )

        def _build_mlp_extractor(self) -> None:
            # Build the network architecture
            self.mlp_extractor = PermEqNetwork(
                self.features_dim,
                act_func=activation_fn,
                arch_obs=POLICY_SIZE_DICT[policy_size],
            )
            
    # ==== Update run params ==== #

    # Ensure reproducability
    init_seed(env_config, exp_config, seed)

    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles,
        train_on_single_scene=exp_config.train_on_single_scene,
    )

    # Set up wandb
    RUN_ID = None
    if exp_config.track_wandb:
        
        # Set up run
        datetime_ = datetime_to_str(dt=datetime.now())
        RUN_ID = f"{exp_config.exp_name}_{datetime_}"
    
        # Add scene to config
        exp_config.scene = env.filename

        run = wandb.init(
            project=sweep_name,
            name=RUN_ID,
            config={**exp_config, **env_config},
            id=RUN_ID,
            **exp_config.wandb,
        )
    
    # Log basic exp info
    logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")
    logging.info(f"Learning in {env_config.num_files} scene(s): {env.env.files}")
    logging.info(f"Obs_space     : {env.observation_space.shape[0]} ---")
    logging.info(f"Action_space\n: {env.env.idx_to_actions}")

    # Set device
    exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
   
    # Load human reference policy
    model = RegularizedPPO(
        reg_policy=human_policy,
        reg_weight=exp_config.reg_weight, # Regularization weight; lambda
        env=env,
        n_steps=exp_config.ppo.n_steps,
        policy=PermEqActorCriticPolicy,
        ent_coef=exp_config.ppo.ent_coef,
        vf_coef=exp_config.ppo.vf_coef,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        verbose=exp_config.verbose,
        tensorboard_log=f"runs/{RUN_ID}" if RUN_ID is not None else None,
        device=exp_config.ppo.device,
    )

    # Learn
    model.learn(
        **exp_config.learn,
        callback=custom_callback,
    )

    # Finish
    if exp_config.track_wandb:
        run.finish()


if __name__ == "__main__":

    # Run
    typer.run(run_hr_ppo)