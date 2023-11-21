import os
import math
import numpy as np
import pandas as pd
import torch
import wandb
import glob
from utils.config import load_config
import torch
from stable_baselines3.common.policies import ActorCriticPolicy
from nocturne.envs.base_env import BaseEnv

class EvaluatePolicy:
    """Evaluate a policy on a set of traffic scenes."""
    
    def __init__(self, env_config, exp_config, policy, eval_files=None, log_to_wandb=True, run=None, deterministic=True, with_replacement=True, file_limit=10):
        self.env_config = env_config
        self.exp_config = exp_config
        self.eval_files = self._get_files(eval_files, file_limit)
        self.policy = policy
        self.log_to_wandb = log_to_wandb
        self.run = run # Wandb run object
        self.deterministic = deterministic
        self.with_replacement = with_replacement
        self.env = BaseEnv(env_config)

        super(EvaluatePolicy).__init__()

    def _get_scores(self):
        """Evaluate policy across a set of traffic scenes."""

        # Make table
        tab_metrics = pd.DataFrame(
            columns=[
                "run_id",
                "traffic_scene",
                "reg_coef",
                "act_acc",
                "act_abs_dist",
                "goal_rate",
                "veh_edge_cr",
                "veh_veh_cr",
                "num_violations",
            ]
        )
            
        for file in self.eval_files:

            # Step through scene in expert control mode to obtain ground truth
            expert_actions, expert_pos, expert_speed, expert_gr, expert_edge_cr, expert_veh_cr = self._step_through_scene(file, expert_mode=True)

            # Step through scene in policy control mode to obtain policy info
            policy_actions, policy_pos, policy_speed, policy_gr, policy_edge_cr, policy_veh_cr = self._step_through_scene(file, expert_mode=False)

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

            action_abs_distance = self.get_action_abs_distance(
                policy_actions, expert_actions, nonnan_ids, self.env.action_space.n
            )

            # Violations of the 3-second rule
            violations_matrix, num_violations = self.get_veh_to_veh_distances(policy_pos, policy_speed)  

            # Store metrics
            scene_perf = {
                "run_id": self.run.id if self.log_to_wandb else None,
                "traffic_scene": file,
                "reg_coef": self.exp_config.reg_weight,
                "act_acc": action_accuracy,
                "act_abs_dist": action_abs_distance,
                "goal_rate": policy_gr,
                "veh_edge_cr": policy_edge_cr,
                "veh_veh_cr": policy_veh_cr,
                "num_violations": num_violations,
            }
            tab_metrics.loc[len(tab_metrics)] = scene_perf        
        
        if self.log_to_wandb:
            wandb.Table(dataframe=tab_metrics)
            self.run.log({"human_metrics": tab_metrics})
            return tab_metrics

        else:
            return tab_metrics

    def _step_through_scene(self, filename: str, expert_mode: bool = False):
        """Step through traffic scene."""

        # Reset simulation
        obs_dict = self.env.reset(filename)
        agent_ids = [agent_id for agent_id in obs_dict.keys()]
        num_steps = self.env_config.episode_length
        num_agents = len(self.env.controlled_vehicles)
        agent_id_to_idx_dict = {agent_id: idx for idx, agent_id in enumerate(agent_ids)} 
        last_info_dicts = {agent_id: {} for agent_id in agent_ids}
        dead_agent_ids = []

        # Storage
        action_indices = np.full(fill_value=np.nan, shape=(num_agents, num_steps))
        agent_positions = np.full(fill_value=np.nan, shape=(num_agents, num_steps, 2))
        agent_speed = np.full(fill_value=np.nan, shape=(num_agents, num_steps))
        goal_achieved, veh_edge_collision, veh_veh_collision = 0, 0, 0

        # Set control mode
        if expert_mode:
            for obj in self.env.controlled_vehicles:
                obj.expert_control = True
        else:
            for obj in self.env.controlled_vehicles:
                obj.expert_control = False
                
        # Step through scene
        for timestep in range(num_steps):   
            
            # Get actions
            if expert_mode:
                for veh_obj in self.env.controlled_vehicles:
                    if veh_obj.id not in dead_agent_ids:
                        # Map actions to nearest grid indices and joint action 
                        expert_action = self.env.scenario.expert_action(veh_obj, timestep)
                        expert_accel, expert_steering, _ = expert_action.numpy()
                        accel_grid_val, _ = self._find_closest_index(self.env.accel_grid, expert_accel)
                        steering_grid_val, _ = self._find_closest_index(self.env.steering_grid, expert_steering)
                        action_idx = self.env.actions_to_idx[accel_grid_val, steering_grid_val][0]
                        veh_idx = agent_id_to_idx_dict[veh_obj.id]

                        # Store action index, position, and speed
                        agent_positions[veh_idx, timestep] = np.array([veh_obj.position.x, veh_obj.position.y])
                        agent_speed[veh_idx, timestep] = veh_obj.speed
                        action_indices[veh_idx, timestep] = action_idx

                action_dict = {} 

            elif self.policy is not None:

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
                    goal_achieved += last_info_dicts[agent_id]["goal_achieved"]
                    veh_edge_collision += last_info_dicts[agent_id]["veh_edge_collision"]
                    veh_veh_collision += last_info_dicts[agent_id]["veh_veh_collision"]
                break

        return (
            action_indices, 
            agent_positions, 
            agent_speed, 
            goal_achieved/num_agents, 
            veh_edge_collision/num_agents, 
            veh_veh_collision/num_agents,
        )
    
    def get_action_accuracy(self, pred_actions, expert_actions, nonnan_ids):
        """Get accuracy of agent actions.
        Args:
            pred_actions: (num_agents, num_steps_per_episode) the predicted actions of the agents.
            expert_actions: (num_agents, num_steps_per_episode) the expert actions of the agents.
            nonnan_ids: (num_agents, num_steps_per_episode) the indices of non-nan actions.
        """
        num_agents = pred_actions.shape[0]
        agg_accuracy = 0

        for idx in range(pred_actions.shape[0]):
            n_samples = pred_actions[0][nonnan_ids[0]].shape[0]
            agent_acc = (pred_actions[idx][nonnan_ids[idx]] == expert_actions[idx][nonnan_ids[idx]]).sum() / n_samples
            agg_accuracy += agent_acc

        return agg_accuracy / num_agents

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

    # Load human reference policy
    saved_variables = torch.load(exp_config.human_policy_path)
    human_policy = ActorCriticPolicy(**saved_variables["data"])
    human_policy.load_state_dict(saved_variables["state_dict"]);
    
    # Evaluate policy
    evaluator = EvaluatePolicy(
        env_config=env_config, 
        exp_config=exp_config,
        policy=human_policy,
        log_to_wandb=False,
    )

    table = evaluator._get_scores()