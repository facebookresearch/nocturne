import copy
import torch
from torch import nn
import torch.nn.functional as F

from box import Box 
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from gymnasium import spaces

from utils.sb3.reg_ppo import RegularizedPPO

class LateFusionNetAttn(nn.Module):
    """Processes the env observation using a late fusion architecture."""

    def __init__(
        self,
        feature_dim: int,
        env_config: Box = None,
        obj_dims: List[int] = [10, 13, 13, 3, 12],
        arch_ego_state: List[int] = [10],
        arch_road_objects: List[int] = [64, 32],
        arch_road_graph: List[int] = [64, 32],
        arch_shared_net: List[int] = [256, 128, 64],
        arch_stop_signs: List[int] = [3],
        act_func: str = "tanh", 
        dropout: float = 0.0,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()    
        # Unpack feature dimensions
        self.ego_input_dim = obj_dims[0]
        self.ro_input_dim = obj_dims[1]
        self.rg_input_dim = obj_dims[2]
        self.ss_input_dim = obj_dims[3]
        self.tl_input_dim = obj_dims[4]

        self.config = env_config
        self._set_obj_dims()

        # Network architectures 
        self.arch_ego_state = arch_ego_state
        self.arch_road_objects = arch_road_objects
        self.arch_road_graph = arch_road_graph
        self.arch_stop_signs = arch_stop_signs  
        self.arch_shared_net = arch_shared_net 
        self.act_func = nn.Tanh() if act_func == "tanh" else nn.ReLU()
        self.dropout = dropout

        # Save output dimensions, used to create the action distribution & value
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
       
        # If using max pool across object dim
        self.shared_net_input_dim = (
            arch_ego_state[-1] +
            arch_road_objects[-1] +
            arch_road_graph[-1] +
            arch_stop_signs[-1]
        )

        # Build the networks 
        # Actor network
        self.actor_ego_state_net = self._build_network(
            input_dim=self.ego_input_dim,
            net_arch=self.arch_ego_state, 
        )
        self.actor_ro_net = self._build_network(
            input_dim=self.ro_input_dim,
            net_arch=self.arch_road_objects, 
        )
        self.actor_ro_attn = nn.MultiheadAttention(
            embed_dim=arch_road_objects[-1], 
            num_heads=1, 
            batch_first=True
        )
        self.actor_rg_net = self._build_network(
            input_dim=self.rg_input_dim,
            net_arch=self.arch_road_graph, 
        )
        self.actor_rg_attn = nn.MultiheadAttention(
            embed_dim=arch_road_objects[-1], 
            num_heads=1, 
            batch_first=True,
        )
        self.actor_ss_net = self._build_network(
            input_dim=self.ss_input_dim,
            net_arch=self.arch_stop_signs,
        )
        self.actor_out_net = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=self.latent_dim_pi,
            net_arch=self.arch_shared_net,
        )

        # Value network
        self.val_ego_state_net = copy.deepcopy(self.actor_ego_state_net) 
        self.val_ro_net = copy.deepcopy(self.actor_ro_net)
        self.val_ro_attn = copy.deepcopy(self.actor_ro_attn)
        self.val_rg_net = copy.deepcopy(self.actor_rg_net)
        self.val_rg_attn = copy.deepcopy(self.actor_rg_attn)
        self.val_ss_net = copy.deepcopy(self.actor_ss_net)
        self.val_out_net = self._build_out_network(
            input_dim=self.shared_net_input_dim,
            output_dim=self.latent_dim_vf,
            net_arch=self.arch_shared_net,
        )
 
    def _build_network(self, input_dim: int, net_arch: List[int],) -> nn.Module:
        """Build a network with the specified architecture."""
        layers = []
        last_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(last_dim, layer_dim))
            layers.append(nn.Dropout(self.dropout))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            last_dim = layer_dim
        return nn.Sequential(*layers)

    def _build_out_network(self, input_dim: int, output_dim: int, net_arch: List[int]):
        """Create the output network architecture."""
        layers = [] 
        prev_dim = input_dim
        for layer_dim in net_arch:
            layers.append(nn.Linear(prev_dim, layer_dim))
            layers.append(nn.LayerNorm(layer_dim))
            layers.append(self.act_func)
            layers.append(nn.Dropout(self.dropout))
            prev_dim = layer_dim
        
        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))
        
        return nn.Sequential(*layers)

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
        """Forward step for the actor network."""
        
        # Unpack observation
        ego_state, road_objects, road_graph, stop_signs = self._unpack_obs(features)    
        
        # Embed features 
        ego_state = self.actor_ego_state_net(ego_state)
        road_objects = self.actor_ro_net(road_objects)
        stop_signs = self.actor_ss_net(stop_signs)
        road_graph = self.actor_rg_net(road_graph)

        # Attention layer
        road_objects, _ = self.actor_ro_attn(road_objects, road_objects, road_objects)
        road_graph, _ = self.actor_rg_attn(road_graph, road_graph, road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(road_objects.permute(0, 2, 1), kernel_size=self.ro_max).squeeze(-1)
        stop_signs = F.max_pool1d(stop_signs.permute(0, 2, 1), kernel_size=self.ss_max).squeeze(-1)
        road_graph = F.max_pool1d(road_graph.permute(0, 2, 1), kernel_size=self.rg_max).squeeze(-1)

        # Concatenate processed ego state and observation and pass through the output layer
        fused = torch.cat((ego_state, road_objects, road_graph, stop_signs), dim=1)
        
        return self.actor_out_net(fused)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """Forward step for the value network."""

        ego_state, road_objects, road_graph, stop_signs = self._unpack_obs(features)

        # Embed features
        ego_state = self.val_ego_state_net(ego_state)
        road_objects = self.val_ro_net(road_objects)
        stop_signs = self.val_ss_net(stop_signs)
        road_graph = self.val_rg_net(road_graph)

        # Attention layer
        road_objects, _ = self.val_ro_attn(road_objects, road_objects, road_objects)
        road_graph, _ = self.val_rg_attn(road_graph, road_graph, road_graph)

        # Max pooling across the object dimension
        # (M, E) -> (1, E) (max pool across features)
        road_objects = F.max_pool1d(road_objects.permute(0, 2, 1), kernel_size=self.ro_max).squeeze(-1)
        stop_signs = F.max_pool1d(stop_signs.permute(0, 2, 1), kernel_size=self.ss_max).squeeze(-1)
        road_graph = F.max_pool1d(road_graph.permute(0, 2, 1), kernel_size=self.rg_max).squeeze(-1)

        # Concatenate processed ego state and observation and pass through the output layer
        fused = torch.cat((ego_state, road_objects, road_graph, stop_signs), dim=1)

        return self.val_out_net(fused)


    def _unpack_obs(self, obs_flat):
        """
        Unpack the flattened observation into the ego state and visible state.
        Args:
            obs_flat (torch.Tensor): flattened observation tensor of shape (batch_size, obs_dim)
        Return:
            ego_state, road_objects, stop_signs, road_graph (torch.Tensor).
        """

        # Unpack ego and visible state
        ego_state = obs_flat[:, :self.ego_input_dim]
        vis_state = obs_flat[:, self.ego_input_dim:]

        # Visible state object order: road_objects, road_points, traffic_lights, stop_signs
        # Find the ends of each section
        ro_end_idx = self.ro_input_dim * self.ro_max
        rg_end_idx = ro_end_idx + (self.rg_input_dim * self.rg_max)
        tl_end_idx = rg_end_idx + (self.tl_input_dim * self.tl_max)
        ss_end_idx = tl_end_idx + (self.ss_input_dim * self.ss_max)
        
        # Unflatten and reshape to (batch_size, num_objects, object_dim)
        road_objects = (vis_state[:, :ro_end_idx]).reshape(-1, self.ro_max, self.ro_input_dim)
        road_graph = (vis_state[:, ro_end_idx:rg_end_idx]).reshape(-1, self.rg_max, self.rg_input_dim,)
        
        # Traffic lights are empty (omitted)
        traffic_lights = (vis_state[:, rg_end_idx:tl_end_idx])    
        stop_signs = (vis_state[:, tl_end_idx:ss_end_idx]).reshape(-1, self.ss_max, self.ss_input_dim)        
        
        return ego_state, road_objects, road_graph, stop_signs
    
    def _set_obj_dims(self):
        # Define original object dimensions
        self.ro_max = self.config.scenario.max_visible_objects
        self.rg_max = self.config.scenario.max_visible_road_points
        self.ss_max = self.config.scenario.max_visible_stop_signs
        self.tl_max = self.config.scenario.max_visible_traffic_lights

class LateFusionAttnPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        env_config: Box,
        mlp_class: Type[LateFusionNetAttn] = LateFusionNetAttn,
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

    # Load configs
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    
    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles,
    )

    obs = env.reset()
    obs = torch.Tensor(obs)[:2]

    # model = LateFusionPermEq(
    #     feature_dim=[
    #         env.ego_state_feat, 
    #         env.road_obj_feat, 
    #         env.road_graph_feat, 
    #         env.stop_sign_feat, 
    #         env.tl_feat],
    #     env_config=env_config
    # )   

    # out = model(obs)
    
    # Define model architecture
    # model_config = Box(
    #     {
    #         "arch_ego_state": [8],
    #         "arch_road_objects": [64],
    #         "arch_road_graph": [128, 64],
    #         "arch_shared_net": [],
    #         "act_func": "tanh",
    #         "dropout": 0.0,
    #         "last_layer_dim_pi": 64,
    #         "last_layer_dim_vf": 64,
    #     }
    # )

    model_config = None

    # Test
    model = RegularizedPPO(
        reg_policy=None,
        reg_weight=None, # Regularization weight; lambda
        env=env,
        n_steps=exp_config.ppo.n_steps,
        policy=LateFusionAttnPolicy,
        ent_coef=exp_config.ppo.ent_coef,
        vf_coef=exp_config.ppo.vf_coef,
        seed=exp_config.seed,  # Seed for the pseudo random generators
        verbose=1,
        device='cuda',
        env_config=env_config,
        mlp_class=LateFusionNetAttn,
        mlp_config=model_config,
    )
    # See architecture
    # print(model.policy)

    model.learn(5000)