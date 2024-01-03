# Multi-agent as vectorized environment
import torch
from torch import nn

from box import Box 
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from utils.sb3.reg_ppo import RegularizedPPO


class LateFusionMLP(nn.Module):
    """
    Custom MLP network for processing multimodal traffic scene observations.
    Args:
        feature_dim (int): dimension of the input feature vector
        env_config (Box): environment configuration
        arch_ego_state (List[int]): architecture of the ego state network
        arch_road_objects (List[int]): architecture of the road objects network
        arch_road_graph (List[int]): architecture of the road graph network
        arch_shared_net (List[int]): architecture of the shared network (after fusion)
        act_func (str): activation function to use
        last_layer_dim_pi (int): dimension of the output layer for the policy network
        last_layer_dim_vf (int): dimension of the output layer for the value network
    """

    def __init__(
        self,
        feature_dim: int,
        env_config: Box,
        arch_ego_state: List[int] = [8],
        arch_road_objects: List[int] = [64],
        arch_road_graph: List[int] = [128, 64],
        arch_shared_net: List[int] = [],
        act_func: str = "tanh", 
        dropout: float = 0.0,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()
        self.config = env_config
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()
        self.arch_ego_state = arch_ego_state
        self.arch_road_objects = arch_road_objects
        self.arch_road_graph = arch_road_graph
        self.arch_shared_net = arch_shared_net
        self.dropout = dropout

        #TODO @Daphne: write function that gets this information from config
        self.input_dim_ego = 10 * self.config.subscriber.n_frames_stacked
        self.input_dim_road_graph = 6500 * self.config.subscriber.n_frames_stacked
        self.input_dim_road_objects = 220 * self.config.subscriber.n_frames_stacked

        # IMPORTANT:Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.shared_net_input_dim = arch_ego_state[-1] + arch_road_objects[-1] + arch_road_graph[-1]

        # POLICY NETWORK
        self.policy_net_ego_state = self._build_ego_state_net(arch_ego_state)
        self.policy_net_road_objects = self._build_road_objects_net(arch_road_objects)
        self.policy_net_road_graph = self._build_road_graph_net(arch_road_graph)
        self.policy_out_layer = self._build_out_net(
            input_dim=self.shared_net_input_dim, 
            output_dim=self.latent_dim_pi,
            net_arch=arch_shared_net
        )
        
        # VALUE NETWORK
        self.val_net_ego_state = self._build_ego_state_net(arch_ego_state)
        self.val_net_road_objects = self._build_road_objects_net(arch_road_objects)
        self.val_net_road_graph = self._build_road_graph_net(arch_road_graph)
        self.val_out_layer = self._build_out_net(
            input_dim=self.shared_net_input_dim,
            output_dim=self.latent_dim_pi,
            net_arch=arch_shared_net
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features (torch.Tensor): input tensor of shape (batch_size, feature_dim)
        Return:
            (torch.Tensor, torch.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the policy network."""
        
        # Unflatten the obs to get the ego state and the visible state items
        ego_state, road_objects, road_points = self._unflatten_obs(features)    
        
        # Process data separately through each network
        ego_state = self.policy_net_ego_state(ego_state)
        road_objects = self.policy_net_road_objects(road_objects)
        road_points = self.policy_net_road_graph(road_points)
        
        # Merge the processed ego state and observation and pass through the output layer
        policy_out = self.policy_out_layer(torch.cat((ego_state, road_objects, road_points), dim=1))
        
        return policy_out

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the value network."""
        # Unflatten the obs to get the ego state and the visible state items
        ego_state, road_objects, road_points = self._unflatten_obs(features)    
        
        # Process data separately through each network
        ego_state = self.val_net_ego_state(ego_state)
        road_objects = self.val_net_road_objects(road_objects)
        road_points = self.val_net_road_graph(road_points)
        
        # Merge the processed ego state and observation and pass through the output layer
        val_out = self.policy_out_layer(torch.cat((ego_state, road_objects, road_points), dim=1))

        return val_out
    

    def _build_ego_state_net(self, net_arch: List[int]):
        """Create ego state network architecture."""
        net_layers = []
        prev_dim = self.input_dim_ego
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net
    
    def _build_road_objects_net(self, net_arch: List[int]):
        """Create road objects architecture."""
        net_layers = []
        prev_dim = self.input_dim_road_objects 
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net

    def _build_road_graph_net(self, net_arch: List[int]):
        """Create road graph network architecture."""
        net_layers = [] 
        prev_dim = self.input_dim_road_graph 
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        net = nn.Sequential(*net_layers)
        return net
    
    def _build_out_net(self, input_dim: int, output_dim: int, net_arch: List[int]):
        """Create the output network architecture."""
        net_layers = [] 
        prev_dim = input_dim
        for layer_dim in net_arch:
            net_layers.append(nn.Linear(prev_dim, layer_dim))
            net_layers.append(nn.Dropout(self.dropout))
            net_layers.append(nn.LayerNorm(layer_dim))
            net_layers.append(self.act_func)
            prev_dim = layer_dim
        
        # Add final layer
        net_layers.append(nn.Linear(prev_dim, output_dim))
        net_layers.append(nn.LayerNorm(output_dim))
        net_layers.append(self.act_func)
        
        # Create network
        net = nn.Sequential(*net_layers)
        return net
    
    def _unflatten_obs(self, obs_flat):
        """Recover indivdiual items in the flattened observation.
        
        Args:
            obs_flat (torch.Tensor): flattened observation tensor
            dim_ego (int): dimension of the ego state

        Returns:
            ego_state (torch.Tensor): ego state tensor
            road_objects (torch.Tensor): road objects tensor
            road_points (torch.Tensor): road points tensor
        """

        # Get ego and visible state
        ego_state, vis_state = obs_flat[:, :self.input_dim_ego], obs_flat[:, self.input_dim_ego:]

        # Visible state object order: road_objects, road_points, traffic_lights, stop_signs
        # Find the ends of each section
        ROAD_OBJECTS_END = (13 * self.config.scenario.max_visible_objects) * self.config.subscriber.n_frames_stacked
        ROAD_POINTS_END = (ROAD_OBJECTS_END + (13 * self.config.scenario.max_visible_road_points)) * self.config.subscriber.n_frames_stacked
        TL_END = (ROAD_POINTS_END + (12 * self.config.scenario.max_visible_traffic_lights)) * self.config.subscriber.n_frames_stacked
        STOP_SIGN_END = (TL_END + (3 * self.config.scenario.max_visible_stop_signs)) * self.config.subscriber.n_frames_stacked
        
        # Unflatten
        road_objects = vis_state[:, :ROAD_OBJECTS_END]
        road_points = vis_state[:, ROAD_OBJECTS_END:ROAD_POINTS_END]
        traffic_lights = vis_state[:, ROAD_POINTS_END:TL_END]
        stop_signs = vis_state[:, TL_END:STOP_SIGN_END]

        return ego_state, torch.hstack((road_objects, stop_signs, traffic_lights)), road_points

class LateFusionMLPPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        env_config: Box,
        mlp_class: Type[LateFusionMLP] = LateFusionMLP,
        mlp_config: Optional[Box] = None,
        *args,
        **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.env_config = env_config
        self.mlp_class = mlp_class
        self.mlp_config = mlp_config if mlp_config is not None else Box({})
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
        self.mlp_extractor = self.mlp_class(
            self.features_dim, 
            self.env_config,
            **self.mlp_config,
        )

if __name__ == "__main__":

    # Load config
    # Load environment and experiment configurations
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")

    env_config.subscriber.n_frames_stacked = 2
    
    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles,

    )

    obs = env.reset()
    obs = torch.Tensor(obs)[:2]


    # Define model architecture
    model_config = Box(
        {
            "arch_ego_state": [8],
            "arch_road_objects": [64],
            "arch_road_graph": [128, 64],
            "arch_shared_net": [],
            "act_func": "tanh",
            "dropout": 0.0,
            "last_layer_dim_pi": 64,
            "last_layer_dim_vf": 64,
        }
    )

    # Test
    model = RegularizedPPO(
        reg_policy=None,
        reg_weight=None, # Regularization weight; lambda
        env=env,
        n_steps=exp_config.ppo.n_steps,
        policy=LateFusionMLPPolicy,
        ent_coef=exp_config.ppo.ent_coef,
        vf_coef=exp_config.ppo.vf_coef,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        verbose=1,
        device='cuda',
        env_config=env_config,
        mlp_class=LateFusionMLP,
        mlp_config=model_config,
    )
    # print(model.policy)
    model.learn(5000)