from box import Box
from stable_baselines3.common.policies import ActorCriticPolicy
import wandb
from gymnasium import spaces
from typing import Callable, Dict
import torch

from networks.mlp_late_fusion import LateFusionMLP
from utils.config import load_config

def load_policy(data_path, file_name, policy_class=ActorCriticPolicy):
    """Load a pretrained policy from a given path."""
    
    checkpoint = torch.load(f"{data_path}/{file_name}.pt")
    policy = policy_class(**checkpoint["data"])
    policy.load_state_dict(checkpoint["state_dict"])
    return policy.eval()
