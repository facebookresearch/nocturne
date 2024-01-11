# Dependencies
import numpy as np
import torch
from stable_baselines3.common.utils import obs_as_tensor

from networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy
from nocturne.envs.base_env import BaseEnv
from utils.config import load_config

MAX_FILES = 100
DET = True

HR_RL_BASE_PATH = f"./models/hr_rl/S{MAX_FILES}"

# Make env
env_config = load_config("env_config")
env_config.num_files = 100
env = BaseEnv(env_config)

# Load policy
policy_name = "nocturne-hr-ppo-01_08_06_34_0.0_S100"

checkpoint = torch.load(f"{HR_RL_BASE_PATH}/{policy_name}.pt")
policy = LateFusionPolicy(
    observation_space=checkpoint["data"]["observation_space"],
    action_space=checkpoint["data"]["action_space"],
    lr_schedule=checkpoint["data"]["lr_schedule"],
    use_sde=checkpoint["data"]["use_sde"],
    env_config=env_config,
    mlp_class=LateFusionNet,
    mlp_config=checkpoint["model_config"],
)
policy.load_state_dict(checkpoint["state_dict"])
# policy.set_training_mode(False)
policy.eval()

total_coll = 0
total_goal_achieved = 0
total_samples = 0

obs_dict = env.reset()
agent_ids = [veh_id for veh_id in obs_dict.keys()]
agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
dead_agent_ids = []
last_info_dicts = {agent_id: {} for agent_id in agent_ids}

for _ in range(5000):
    # Get actions
    action_dict = {}
    for agent_id in obs_dict:
        agent_idx = agent_id_to_idx[agent_id]
        # Get observation
        obs = torch.from_numpy(obs_dict[agent_id]).unsqueeze(dim=0)
        # Get action
        action, _ = policy.predict(obs, deterministic=DET)
        # Store action
        action_dict[agent_id] = int(action)

    # Step in the environment
    obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

    for agent_id, is_done in done_dict.items():
        if is_done and agent_id not in dead_agent_ids:
            dead_agent_ids.append(agent_id)
            # Store agents' last info dict
            last_info_dicts[agent_id] = info_dict[agent_id].copy()

    if done_dict["__all__"]:
        # Update stats
        for agent_id in agent_ids:
            total_coll += last_info_dicts[agent_id]["collided"] * 1
            total_goal_achieved += last_info_dicts[agent_id]["goal_achieved"] * 1

        # num episodes x num agents
        total_samples += len(agent_ids)

        # Reset
        obs_dict = env.reset()
        agent_ids = [veh_id for veh_id in obs_dict.keys()]
        agent_id_to_idx = {agent_id: idx for idx, agent_id in enumerate(agent_ids)}
        dead_agent_ids = []
        last_info_dicts = {agent_id: {} for agent_id in agent_ids}

        print(f"Reset to scene: {env.file} with {len(agent_ids)} agents")

print(f"num_coll: {total_coll} | num_goal_achieved: {total_goal_achieved} | num_samples: {total_samples}")
print(f"goal_rate: {(total_goal_achieved/total_samples)*100:.2f} %")
print(f"coll_rate: {(total_coll/total_samples)*100:.2f} %")
