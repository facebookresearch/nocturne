# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces
import torch
from torch import nn

from utils.sb3.reg_ppo import RegularizedPPO


class PermEqNetwork(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        arch_ego_state: List[int] = [8],
        arch_obs: List[int] = [126, 64],
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.ego_state_dim = 10 # Always has 10 elements 
        self.observation_dim = feature_dim - self.ego_state_dim

        # IMPORTANT:Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # POLICY NETWORK
        self.policy_net_ego_state = self._build_ego_state_net(arch_ego_state)
        self.policy_net_obs = self._build_obs_net(arch_obs)
        self.policy_out_layer = nn.Sequential(
            nn.Linear(arch_ego_state[-1] + arch_obs[-1], self.latent_dim_pi),
            nn.Tanh(),
        )
        # VALUE NETWORK
        self.value_net_ego_state = self._build_ego_state_net(arch_ego_state)
        self.value_net_obs = self._build_obs_net(arch_obs)
        self.value_out_layer = nn.Sequential(
            nn.Linear(arch_ego_state[-1] + arch_obs[-1], self.latent_dim_vf),
            nn.Tanh(),
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :return: (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the policy network."""
        # Partition the input into the ego state and the observations
        ego_state, obs = features[:, :10], features[:, 10:]

        # Process the ego state and observation separately
        policy_ego_processed = self.policy_net_ego_state(ego_state)
        policy_obs_processed = self.policy_net_obs(obs)

        # Merge the processed ego state and observation and pass through the output layer
        policy_out = self.policy_out_layer(torch.cat((policy_ego_processed, policy_obs_processed), dim=1))
        
        return policy_out

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the value network."""

        # Partition the input into the ego state and the observations
        ego_state, obs = features[:, :10], features[:, 10:]

        # Process the ego state and observation separately
        val_ego_processed = self.value_net_ego_state(ego_state)
        val_obs_processed = self.value_net_obs(obs)

        # Merge the processed ego state and observation and pass through the output layer
        val_out = self.value_out_layer(torch.cat((val_ego_processed, val_obs_processed), dim=1))
        
        return val_out
    
    def _build_ego_state_net(self, net_arch: List[int]):
        """Create ego state network architecture."""
        net_layers = []
        prev_dim = self.ego_state_dim # Initial dimension for concatenation
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Tanh())
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net

    def _build_obs_net(self, net_arch: List[int]):
        """Create obstructed view network architecture."""
        net_layers = []
        prev_dim = self.observation_dim # Initial dimension for concatenation
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Tanh())
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net

class PEActorCriticPolicy(ActorCriticPolicy):
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
        self.mlp_extractor = PermEqNetwork(
            self.features_dim,
            arch_ego_state=[8, 4, 2],
            arch_obs=[500, 200, 50],
        )


if __name__ == "__main__":

    # Load config
    # Load environment and experiment configurations
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    
    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles,
        train_on_single_scene=exp_config.train_on_single_scene,
    )

    obs = env.reset()
    obs = torch.Tensor(obs)[:2]

    # Make model
    net = PermEqNetwork(
        feature_dim=obs.shape[1], 
        last_layer_dim_pi=64, 
        last_layer_dim_vf=64
    )

    net(obs)

    # Test
    model = RegularizedPPO(
        reg_policy=None,
        reg_weight=None, # Regularization weight; lambda
        env=env,
        n_steps=exp_config.ppo.n_steps,
        policy=PEActorCriticPolicy,
        ent_coef=exp_config.ppo.ent_coef,
        vf_coef=exp_config.ppo.vf_coef,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        verbose=1,
        device='cuda',
    )
    print(model.policy)
    model.learn(5000)