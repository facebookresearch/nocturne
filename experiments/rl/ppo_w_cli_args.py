import typer
import numpy as np
from utils.config import load_config
from typing import List

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
    policy_layers: List[int]=[64, 64]
) -> None:

    """Train RL agent using PPO with CLI arguments."""

    print(f'steer_disc: {steer_disc} | accel_disc: {accel_disc} | ent_coef: {ent_coef}')


if __name__ == "__main__":

    # Run
    typer.run(run_ppo)