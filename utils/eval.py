import os
import math
import numpy as np
import pandas as pd
import torch
import wandb
import glob
from utils.config import load_config
import torch
from utils.policies import load_policy
from stable_baselines3.common.policies import ActorCriticPolicy
from nocturne.envs.base_env import BaseEnv

class EvaluatePolicy:
    """Evaluate a policy on a set of traffic scenes."""
    
    def __init__(
        self, 
        env_config, 
        exp_config, 
        policy, 
        eval_files=None, 
        baselines=['random', 'expert'], 
        reg_coef=None, 
        log_to_wandb=True, 
        run=None, 
        deterministic=True, 
        with_replacement=True, 
        return_trajectories=False,
        file_limit=10
        ):
        self.env_config = env_config
        self.exp_config = exp_config
        self.policy = policy
        self.eval_files = self._get_files(eval_files, file_limit)
        self.baselines = baselines
        self.reg_coef = reg_coef
        self.log_to_wandb = log_to_wandb
        self.run = run # Wandb run object
        self.deterministic = deterministic
        self.with_replacement = with_replacement
        self.return_trajectories = return_trajectories
        self.env = BaseEnv(env_config)

        super(EvaluatePolicy).__init__()

    def _get_scores(self):
        """Evaluate policy across a set of traffic scenes."""

        # Make tables
        df_eval = pd.DataFrame(
            columns=[
                "run_id",
                "traffic_scene",
                "agents_controlled",
                "reg_coef",
                "act_acc",
                "pos_rmse",
                "speed_mae",
                "goal_rate",
                "veh_edge_cr",
                "veh_veh_cr",
                "num_violations",
            ]
        )

        df_trajs = pd.DataFrame(
            columns=[
                "traffic_scene",
                "timestep",
                "agent_id",
                "policy_pos_x", 
                "policy_pos_y",
                "policy_speed",
                "policy_act",
                "expert_pos_x",
                "expert_pos_y",
                "expert_speed",
                "expert_act",
            ]
        )
            
        for file in self.eval_files:
            
            # Make sure the agent ids are in the same order (currently can't make copies of the env)
            obs_dict = self.env.reset(file)
            self.agent_ids = [agent_id for agent_id in obs_dict.keys()]

            # Step through scene in expert control mode to obtain ground truth
            expert_actions, expert_pos, expert_speed, expert_gr, expert_edge_cr, expert_veh_cr = self._step_through_scene(
                file, mode="expert", 
            )

            # Step through scene in policy control mode to obtain policy info
            policy_actions, policy_pos, policy_speed, policy_gr, policy_edge_cr, policy_veh_cr = self._step_through_scene(
                file, mode="policy"
            )

            # Filter out invalid steps 
            nonnan_ids = np.logical_not(
                np.logical_or(
                    np.isnan(policy_actions),
                    np.isnan(expert_actions),
                )
            )
            # Compute metrics
            action_accuracy = self.get_action_accuracy(
                policy_actions, expert_actions, nonnan_ids
            )

            position_rmse = self.get_pos_rmse(
                policy_pos, expert_pos, nonnan_ids
            )

            speed_agent_mae = self.get_speed_mae(
                policy_speed, expert_speed, nonnan_ids
            )
        
            # Violations of the 3-second rule
            violations_matrix, num_violations = self.get_veh_to_veh_distances(policy_pos, policy_speed)  

            # Store metrics
            scene_perf = {
                "run_id": self.run.id if self.log_to_wandb else None,
                "traffic_scene": file,
                "agents_controlled": expert_actions.shape[0],
                "reg_coef": self.exp_config.reg_weight if self.reg_coef is None else self.reg_coef,
                "act_acc": action_accuracy,
                "pos_rmse": position_rmse,
                "speed_mae": speed_agent_mae,
                "goal_rate": policy_gr,
                "veh_edge_cr": policy_edge_cr,
                "veh_veh_cr": policy_veh_cr,
                "num_violations": num_violations,
            }
            df_eval.loc[len(df_eval)] = scene_perf 
            
            if self.return_trajectories:
                scene_trajs = pd.DataFrame({
                    "traffic_scene": file, #TODO: repeat 
                    "timestep": np.tile(list(range(self.env_config.episode_length)), self.num_agents), 
                    "agent_id": np.repeat(list(range(self.num_agents)), self.env_config.episode_length),
                    "policy_pos_x": policy_pos[:, :, 0].flatten(), # num_agents * 80
                    "policy_pos_y": policy_pos[:, :, 1].flatten(),
                    "policy_speed": policy_speed.flatten(),
                    "policy_act": policy_actions.flatten(),
                    "expert_pos_x": expert_pos[:, :, 0].flatten(),
                    "expert_pos_y": expert_pos[:, :, 1].flatten(),
                    "expert_act": expert_actions.flatten(),
                    "expert_speed": expert_speed.flatten(),
                })
                df_trajs = pd.concat([df_trajs, scene_trajs], ignore_index=True)
           
        if self.log_to_wandb:
            wandb.Table(dataframe=df_eval)
            self.run.log({"human_metrics": df_eval})
            return df_eval

        else:
            if self.return_trajectories:
                return df_eval, df_trajs
            else:
                return df_eval


    def _step_through_scene(self, filename: str, mode: str):
        """Step through traffic scene.
        Args:
            filename: (str) the name of the traffic scene file.
            mode: (str) the control mode of the agents.
        """
        # Reset env
        obs_dict = self.env.reset(filename)
        num_steps = self.env_config.episode_length
        self.num_agents = len(self.env.controlled_vehicles)
        agent_id_to_idx_dict = {agent_id: idx for idx, agent_id in enumerate(self.agent_ids)} 
        last_info_dicts = {agent_id: {} for agent_id in self.agent_ids}
        dead_agent_ids = []
        
        # Storage
        action_indices = np.full(fill_value=np.nan, shape=(self.num_agents, num_steps))
        agent_positions = np.full(fill_value=np.nan, shape=(self.num_agents, num_steps, 2))
        agent_speed = np.full(fill_value=np.nan, shape=(self.num_agents, num_steps))
        goal_achieved, veh_edge_collision, veh_veh_collision = 0, 0, 0

        # Set control mode
        if mode == "expert":
            for obj in self.env.controlled_vehicles:
                obj.expert_control = True
        if mode == "policy":
            for obj in self.env.controlled_vehicles:
                obj.expert_control = False
                
        # Step through scene
        for timestep in range(num_steps):   
            
            # Get actions
            if mode == "expert":
                for veh_obj in self.env.controlled_vehicles:
                    if veh_obj.id not in dead_agent_ids:
                        # Map actions to nearest grid indices and joint action 
                        expert_action = self.env.scenario.expert_action(veh_obj, timestep)
                        if expert_action is not None:
                            expert_accel, expert_steering, _ = expert_action.numpy()
                            accel_grid_val, _ = self._find_closest_index(self.env.accel_grid, expert_accel)
                            steering_grid_val, _ = self._find_closest_index(self.env.steering_grid, expert_steering)
                            action_idx = self.env.actions_to_idx[accel_grid_val, steering_grid_val][0]
                            veh_idx = agent_id_to_idx_dict[veh_obj.id]

                            # Store action index, position, and speed
                            agent_positions[veh_idx, timestep] = np.array([veh_obj.position.x, veh_obj.position.y])
                            agent_speed[veh_idx, timestep] = veh_obj.speed
                            action_indices[veh_idx, timestep] = action_idx
                        # else:
                        #     #print(f'veh {veh_obj.id} at t = {timestep} returns None action!')

                action_dict = {} 

            elif mode == "policy" and self.policy is not None:

                observations = self._obs_dict_to_tensor(obs_dict)

                actions, _ = self.policy.predict(
                    observations,
                    deterministic=self.deterministic,
                )

                action_dict = dict(zip(obs_dict.keys(), actions))

                for veh_obj in self.env.controlled_vehicles:
                    if veh_obj.id not in dead_agent_ids:
                        veh_idx = agent_id_to_idx_dict[veh_obj.id]
                        agent_positions[veh_idx, timestep] = np.array([veh_obj.position.x, veh_obj.position.y])
                        agent_speed[veh_idx, timestep] = veh_obj.speed
                        action_indices[veh_idx, timestep] = action_dict[veh_obj.id]

            elif mode == "random":
                actions = np.random.randint(0, self.env.num_actions, size=(self.num_agents,))
                    
            # Step env
            obs_dict, rew_dict, done_dict, info_dict = self.env.step(action_dict)

            # Update dead agents based on most recent done_dict
            for agent_id, is_done in done_dict.items():
                if is_done and agent_id not in dead_agent_ids:
                    dead_agent_ids.append(agent_id)
            
                    # Store agents' last info dict
                    last_info_dicts[agent_id] = info_dict[agent_id].copy()

            if done_dict["__all__"]:
                for agent_id in self.agent_ids:
                    goal_achieved += last_info_dicts[agent_id]["goal_achieved"]
                    veh_edge_collision += last_info_dicts[agent_id]["veh_edge_collision"]
                    veh_veh_collision += last_info_dicts[agent_id]["veh_veh_collision"]
                break

        return (
            action_indices, 
            agent_positions, 
            agent_speed, 
            goal_achieved/self.num_agents, 
            veh_edge_collision/self.num_agents, 
            veh_veh_collision/self.num_agents,
        )
    
    def get_action_accuracy(self, pred_actions, expert_actions, nonnan_ids):
        """Get accuracy of agent actions.
        Args:
            pred_actions: (num_agents, num_steps_per_episode) the predicted actions of the agents.
            expert_actions: (num_agents, num_steps_per_episode) the expert actions of the agents.
            nonnan_ids: (num_agents, num_steps_per_episode) the indices of non-nan actions.
        """
        return (expert_actions[nonnan_ids] == pred_actions[nonnan_ids]).sum() / nonnan_ids.flatten().shape[0]

    def get_pos_rmse(self, pred_actions, expert_actions, nonnan_ids):
        return np.sqrt(np.linalg.norm(pred_actions[nonnan_ids] - expert_actions[nonnan_ids])).mean()
    
    def get_speed_mae(self, pred_actions, expert_actions, nonnan_ids):
        return np.abs(pred_actions[nonnan_ids] - expert_actions[nonnan_ids]).mean()
    
    def get_steer_mae(self, pred_actions, expert_actions, nonnan_ids):
        return np.abs(pred_actions[nonnan_ids] - expert_actions[nonnan_ids]).mean()
    
    def get_action_abs_distance(self, pred_actions, expert_actions, nonnan_ids, action_space_dim):
        """Get accuracy of agent actions.
        Args:
            pred_actions: (num_agents, num_steps_per_episode) the predicted actions of the agents.
            expert_actions: (num_agents, num_steps_per_episode) the expert actions of the agents.
            nonnan_ids: (num_agents, num_steps_per_episode) the indices of non-nan actions.
        """

        num_agents = pred_actions.shape[0]
        agg_abs_dist = 0

        for idx in range(pred_actions.shape[0]):
            n_samples = pred_actions[0][nonnan_ids[0]].shape[0]
            agent_abs_dist = np.abs(pred_actions[idx][nonnan_ids[idx]] - expert_actions[idx][nonnan_ids[idx]]).sum() / n_samples
            agg_abs_dist += agent_abs_dist

        return agg_abs_dist / num_agents

    def get_veh_to_veh_distances(self, positions, velocities, time_gap_in_sec=3):
        """Calculate distances between vehicles at each time step and track 
            whether the 3-second rule is violated. Rule of thumb for safe driving: 
            allow at least 3 seconds between you and the car in front of you. 
        """
        num_steps = 80

        num_vehicles = positions.shape[0]
        veh_distances_per_step = np.zeros((num_vehicles, num_vehicles, num_steps))
        safe_distances_per_step = np.zeros((num_vehicles, num_vehicles, num_steps))

        for step in range(num_steps):
            for veh_i in range(num_vehicles):
                for veh_j in range(num_vehicles):
                    if veh_i == veh_j or np.isnan(positions[:, step]).any():
                        continue # Skip nans
                    else:
                        # Compute distance from veh_i to to the other vehicles
                        # One step is 0.1 seconds, speed is in mph
                        distance_between_veh_ij = math.sqrt((positions[veh_i, step][0] - positions[veh_j, step][0])**2 + (positions[veh_i, step][1] - positions[veh_j, step][1])**2)
                        safe_distance = velocities[veh_i, step] * (time_gap_in_sec / 3600) # Convert time gap from seconds to hours

                        # Store
                        safe_distances_per_step[veh_i, veh_j, step] = safe_distance
                        veh_distances_per_step[veh_i, veh_j, step] = distance_between_veh_ij

                        if distance_between_veh_ij < safe_distance:
                            print(f"Vehicles {veh_i + 1} and {veh_j + 1} are too close!")

        # Aggregate
        distance_violations_matrix = (veh_distances_per_step < safe_distances_per_step).sum(axis=2)
                
        return distance_violations_matrix, distance_violations_matrix.sum()     

    def _find_closest_index(self, action_grid, action):
        """Find the nearest value in the action grid for a given expert action."""
        indx = np.argmin(np.abs(action_grid - action))
        return action_grid[indx], indx

    def _obs_dict_to_tensor(self, obs_dict):
        """Convert obs dict to tensor."""
        obs = torch.zeros(size=(len(obs_dict.keys()), self.env.observation_space.shape[0]))
        for idx, agent_id in enumerate(obs_dict.keys()):
            obs[idx, :] = torch.Tensor(obs_dict[agent_id])
        return obs

    def _get_files(self, eval_files, file_limit):
        "Extract traffic scenes for evaluation."
        if eval_files is not None:
            return eval_files
        else:
            file_paths = glob.glob(self.env_config.data_path + "/tfrecord*")
            eval_files = [os.path.basename(file) for file in file_paths][:file_limit]
            return eval_files


if __name__ == "__main__":

    env_config = load_config("env_config")
    exp_config = load_config("exp_config")

    # Load trained human reference policy
    human_policy = load_policy(
        data_path="./models/il",
        file_name="human_policy_single_scene_2023_11_22",   
    )

    # Evaluate policy
    evaluator = EvaluatePolicy(
        env_config=env_config, 
        exp_config=exp_config,
        policy=human_policy,
        log_to_wandb=False,
        deterministic=True,
        reg_coef=0.0,
        return_trajectories=True,
    )

    il_results_check = evaluator._get_scores()