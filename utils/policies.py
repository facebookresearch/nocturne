from stable_baselines3.common.policies import ActorCriticPolicy
import wandb
import torch

def load_policy(data_path, file_name, policy_class=ActorCriticPolicy):
    """Load a pretrained policy from a given path."""
    
    checkpoint = torch.load(f"{data_path}/{file_name}.pt")
    policy = policy_class(**checkpoint["data"])
    policy.load_state_dict(checkpoint["state_dict"])
    return policy.eval()

def load_rl_policy(
    run,
):
    pass
    #TODO @Daphne: implement this