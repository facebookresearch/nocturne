import typer
from typing import Tuple, Optional

import typer
from typing_extensions import Annotated

import numpy as np
from utils.config import load_config
from typing import List
import wandb

# Load environment and experiment configurations
env_config = load_config("env_config")
exp_config = load_config("exp_config")
video_config = load_config("video_config")

def run_ppo(
    steer_disc: int=5, 
    accel_disc: int=5, 
    ent_coef: float=0.,
    vf_coef: float=0.5,
    seed: int=42,
    policy_layers: Annotated[Optional[str], typer.Argument()] = [512, 256, 128]
) -> None:
    """Train RL agent using PPO with CLI arguments."""

    breakpoint()

    print(f'steer_disc: {steer_disc} | accel_disc: {accel_disc} | ent_coef: {ent_coef}')


if __name__ == "__main__":

    # Run
    typer.run(run_ppo)