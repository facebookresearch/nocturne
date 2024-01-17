import torch
from tqdm import tqdm

from nocturne.envs.base_env import BaseEnv
from utils.config import load_config
from utils.policies import load_policy


def evaluate_policy(
    env_config, mode, policy=None, num_files=1000, controlled_agents=1, total_steps=1000, deterministic=True
):
    # Set the number of vehicles to control per scene
    env_config.max_num_vehicles = controlled_agents
    # Set which files to use
    env_config.num_files = num_files

    # Make env
    env = BaseEnv(env_config)

    # Storage
    total_off_road = 0
    total_coll = 0
    total_goal_achieved = 0
    total_samples = 0
    num_episodes = 0

    # Run
    obs_dict = env.reset()
    agent_ids = [veh_id for veh_id in obs_dict.keys()]
    dead_agent_ids = []
    last_info_dicts = {agent_id: {} for agent_id in agent_ids}

    for _ in tqdm(range(total_steps)):
        # Get actions
        action_dict = {}

        if mode == "policy" and policy is not None:
            for agent_id in obs_dict:
                # Get observation
                obs = torch.from_numpy(obs_dict[agent_id]).unsqueeze(dim=0)

                # Get action
                action, _ = policy.predict(obs, deterministic=deterministic)
                action_dict[agent_id] = int(action)

        elif mode == "expert_replay" or policy is None:
            # Use expert actions
            for veh in env.controlled_vehicles:
                veh.expert_control = True

        # Take a step
        obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

        for agent_id, is_done in done_dict.items():
            if is_done and agent_id not in dead_agent_ids:
                dead_agent_ids.append(agent_id)
                # Store agents' last info dict
                last_info_dicts[agent_id] = info_dict[agent_id].copy()

        if done_dict["__all__"]:
            # Update stats
            for agent_id in agent_ids:
                total_coll += last_info_dicts[agent_id]["veh_veh_collision"] * 1
                total_off_road += last_info_dicts[agent_id]["veh_edge_collision"] * 1
                total_goal_achieved += last_info_dicts[agent_id]["goal_achieved"] * 1

            # total_samples = num episodes x num agents
            total_samples += len(agent_ids)
            num_episodes += 1

            # Reset
            obs_dict = env.reset()
            agent_ids = [veh_id for veh_id in obs_dict.keys()]
            dead_agent_ids = []
            last_info_dicts = {agent_id: {} for agent_id in agent_ids}

    # Compute metrics
    goal_rate = (total_goal_achieved / total_samples) * 100
    off_road = (total_off_road / total_samples) * 100
    coll_rate = (total_coll / total_samples) * 100

    # print(f"\n ---- Performance ---- {mode}: \n")
    # print(f"num_coll: {total_coll} | num_goal_achieved: {total_goal_achieved} | (num_episodes * num_agents): {total_samples} \n")
    # print(f"goal_rate: {(total_goal_achieved/total_samples)*100:.2f} %")
    # print(f"off_road: {(total_off_road/total_samples)*100:.2f} %")
    # print(f"coll_rate: {(total_coll/total_samples)*100:.2f} % \n")

    return goal_rate, off_road, coll_rate, total_samples


if __name__ == "__main__":
    BASE_PATH = "./models/il"
    BC_POLICY_NAME = "human_policy_S1000_01_12_11_11"
    TOTAL_STEPS = 80_000

    env_config = load_config("env_config")

    # Load human policy
    human_policy = load_policy(
        data_path=BASE_PATH,
        file_name=BC_POLICY_NAME,
    )

    # Evaluate human policy
    evaluate_policy(
        env_config=env_config, mode="il_policy", policy=human_policy, controlled_agents=1, total_steps=TOTAL_STEPS
    )

    # Expert replay
    evaluate_policy(
        env_config=env_config, mode="expert_replay", policy=None, controlled_agents=1, total_steps=TOTAL_STEPS
    )
