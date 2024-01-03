import os
import math
import numpy as np
import pandas as pd
import logging
from tqdm import tqdm
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
        file_limit=1000
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

        logging.info(f'\n Evaluating policy on {len(self.eval_files)} files...')

        # Create tables
        df_eval = pd.DataFrame(  
            columns=[
                "run_id",
                "reg_coef",
                "traffic_scene",
                "agent_id",
                "act_acc",
                "accel_val_mae",
                "steer_val_mae",
                "pos_rmse",
                "speed_mae",
                "goal_rate",
                "veh_edge_cr",
                "veh_veh_cr",
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
            
        for file in tqdm(self.eval_files):

            logging.debug(f"Evaluating policy on {file}...")
            
            # Step through scene in expert control mode to obtain ground truth
            expert_actions, expert_pos, expert_speed, expert_gr, expert_edge_cr, expert_veh_cr = self._step_through_scene(
                file, mode="expert", 
            )

            # Step through scene in policy control mode to obtain policy info
            policy_actions, policy_pos, policy_speed, policy_gr, policy_edge_cr, policy_veh_cr = self._step_through_scene(
                file, mode="policy"
            )

            # Filter out invalid steps 
            nonnan_ids = ~np.isnan(expert_actions)

            # Compute metrics
            action_accuracies = self.get_action_accuracy(
                policy_actions, expert_actions, nonnan_ids
            )

            position_rmse = self.get_pos_rmse(
                policy_pos, expert_pos, 
            )

            speed_agent_mae = self.get_speed_mae(
                policy_speed, expert_speed,
            )

            abs_diff_accel, abs_diff_steer = self.get_action_val_diff(
                policy_actions, expert_actions,
            )
        
            # Violations of the 3-second rule
            #violations_matrix, num_violations = self.get_veh_to_veh_distances(policy_pos, policy_speed)  

            # Store metrics
            scene_perf = pd.DataFrame({
                "run_id": self.run.id if self.log_to_wandb else None,
                "reg_coef": np.repeat(self.reg_coef, len(self.agent_names)),
                "traffic_scene": file,
                "agent_id": self.agent_names,
                "act_acc": action_accuracies,
                "accel_val_mae": abs_diff_accel,
                "steer_val_mae": abs_diff_steer,
                "pos_rmse": position_rmse,
                "speed_mae": speed_agent_mae,
                "goal_rate": policy_gr,
                "veh_edge_cr": policy_edge_cr,
                "veh_veh_cr": policy_veh_cr,
            })
            df_eval = pd.concat([df_eval, scene_perf], ignore_index=True)
            
            if self.return_trajectories:
                scene_trajs = pd.DataFrame({
                    "traffic_scene": file,  
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

        # Make sure the agent ids are in the same order
        agent_ids = np.sort([veh.id for veh in self.env.controlled_vehicles])
        self.agent_names = agent_ids
        agent_id_to_idx_dict = {agent_id: idx for idx, agent_id in enumerate(agent_ids)} 
        last_info_dicts = {agent_id: {} for agent_id in agent_ids}
        dead_agent_ids = []
        
        # Storage
        action_indices = np.full(fill_value=np.nan, shape=(self.num_agents, num_steps))
        agent_positions = np.full(fill_value=np.nan, shape=(self.num_agents, num_steps, 2))
        agent_speed = np.full(fill_value=np.nan, shape=(self.num_agents, num_steps))
        goal_achieved, veh_edge_collision, veh_veh_collision = np.zeros(self.num_agents), np.zeros(self.num_agents), np.zeros(self.num_agents)

        # Set control mode
        if mode == "expert":
            logging.debug(f'EXPERT MODE')
            for obj in self.env.controlled_vehicles:
                obj.expert_control = True
        if mode == "policy":
            logging.debug(f'POLICY MODE')
            for obj in self.env.controlled_vehicles:
                obj.expert_control = False
        
        logging.debug(f'agent_ids: {agent_ids}')
        
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
                        else:
                            # Skip None actions (these are invalid)
                            logging.debug(f'veh {veh_obj.id} at t = {timestep} returns None action!')
                            continue                        

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
                for agent_id in agent_ids:
                    agent_idx = agent_id_to_idx_dict[agent_id]
                    goal_achieved[agent_idx] += last_info_dicts[agent_id]["goal_achieved"]
                    veh_edge_collision[agent_idx] += last_info_dicts[agent_id]["veh_edge_collision"]
                    veh_veh_collision[agent_idx] += last_info_dicts[agent_id]["veh_veh_collision"]
                break

        return (
            action_indices, 
            agent_positions, 
            agent_speed, 
            goal_achieved,
            veh_edge_collision,
            veh_veh_collision,
        )

    def get_action_val_diff(self, pred_actions, expert_actions):
        """Get difference between human action values and predicted action values.
        Args:
            pred_actions: (num_agents, num_steps_per_episode) the predicted actions of the agents.
            expert_actions: (num_agents, num_steps_per_episode) the expert actions of the agents.
            nonnan_ids: (num_agents, num_steps_per_episode) the indices of non-nan actions.
        """
        # Filter out invalid actions 
        nonnan_ids = np.logical_not(
            np.logical_or(
                np.isnan(pred_actions),
                np.isnan(expert_actions),
            )
        )
        num_agents = expert_actions.shape[0]
        arr = np.zeros((num_agents, 2))
        for agent_idx in range(num_agents):
            not_nan = nonnan_ids[agent_idx, :]

            valid_expert_acts = expert_actions[agent_idx, :][not_nan]
            valid_pred_acts = pred_actions[agent_idx, :][not_nan]

            exp_acc_vals, exp_steer_vals = np.zeros_like(valid_pred_acts), np.zeros_like(valid_pred_acts)
            pred_acc_vals, pred_steer_vals = np.zeros_like(valid_pred_acts), np.zeros_like(valid_pred_acts)

            for idx in range(valid_expert_acts.shape[0]):
                # Get expert and predicted values
                exp_acc_vals[idx], exp_steer_vals[idx] = self.env.idx_to_actions[valid_expert_acts[idx]]
                pred_acc_vals[idx], pred_steer_vals[idx] = self.env.idx_to_actions[valid_pred_acts[idx]]

            # Get mean absolute difference
            abs_accel_diff = np.abs(exp_acc_vals - pred_acc_vals).mean()
            abs_steer_diff = np.abs(exp_steer_vals - pred_steer_vals).mean()

            # Store
            arr[agent_idx, 0] = abs_accel_diff
            arr[agent_idx, 1] = abs_steer_diff

        return arr[:, 0], arr[:, 1] # abs_diff_accel, abs_diff_steer
    
    def get_action_accuracy(self, pred_actions, expert_actions, nonnan_ids):
        """Get accuracy of agent actions.
        Args:
            pred_actions: (num_agents, num_steps_per_episode) the predicted actions of the agents.
            expert_actions: (num_agents, num_steps_per_episode) the expert actions of the agents.
            nonnan_ids: (num_agents, num_steps_per_episode) the indices of non-nan actions.
        """
        num_agents = expert_actions.shape[0]
        arr = np.zeros(num_agents)
        for agent_idx in range(num_agents):
            not_nan = nonnan_ids[agent_idx, :]
            arr[agent_idx] = (expert_actions[agent_idx, :][not_nan] == pred_actions[agent_idx, :][not_nan]).sum() / not_nan.shape[0]
        return arr

    def get_pos_rmse(self, pred_actions, expert_actions):
        
        # Filter out invalid actions 
        nonnan_ids = np.logical_not(
            np.logical_or(
                np.isnan(pred_actions),
                np.isnan(expert_actions),
            )
        )
        num_agents = expert_actions.shape[0]
        arr = np.zeros(num_agents)
        for agent_idx in range(num_agents):
            not_nan = nonnan_ids[agent_idx, :]
            arr[agent_idx] = (np.sqrt(np.linalg.norm(pred_actions[agent_idx, :][not_nan] - expert_actions[agent_idx, :][not_nan]))).mean()
        return arr
    
    def get_speed_mae(self, pred_actions, expert_actions):
        # Filter out invalid actions 
        nonnan_ids = np.logical_not(
            np.logical_or(
                np.isnan(pred_actions),
                np.isnan(expert_actions),
            )
        )
        num_agents = expert_actions.shape[0]
        arr = np.zeros(num_agents)
        for agent_idx in range(num_agents):
            not_nan = nonnan_ids[agent_idx, :]
            arr[agent_idx] = (np.abs(pred_actions[agent_idx, :][not_nan] - expert_actions[agent_idx, :][not_nan])).mean()
        return arr

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
                            logging.debug(f"Vehicles {veh_i + 1} and {veh_j + 1} are too close!")

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

    MAX_FILES = 50

    # Train
    train_file_paths = glob.glob(f"{env_config.data_path}" + "/tfrecord*")
    train_eval_files = [os.path.basename(file) for file in train_file_paths][:MAX_FILES]

    # Load human reference policy
    human_policy = load_policy(
        data_path="./models/il",
        file_name="human_policy_2_scenes_2023_11_22",   
    )

    # Evaluate policy
    evaluator = EvaluatePolicy(
        env_config=env_config, 
        exp_config=exp_config,
        policy=human_policy,
        eval_files=train_eval_files,
        log_to_wandb=False,
        deterministic=True,
        reg_coef=0.0,
        return_trajectories=True,
    )

    df_il_res_2, df_il_trajs_2 = evaluator._get_scores()