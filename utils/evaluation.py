import numpy as np
import wandb
from pyvirtualdisplay import Display

from utils.render import make_video

def evaluate_policy(
    model,
    env,
    n_steps_per_episode=80,
    n_eval_episodes=1,
    eval_files=None,
    eval_modes=['expert', 'policy'],
    deterministic = True,
    render = False,
    video_caption = None,
    video_config = None,
    return_episode_rewards = False,
    verbose=1,
):
    """
    Runs policy for ``n_eval_episodes`` episodes and returns average reward.

    Args:
    -----
    model: The IL/RL policy to evaluate. This can be any object that implements 
        a `predict` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    env: The gym environment or ``VecEnv`` environment.
    n_eval_episodes: Number of different traffic scenes in which to evaluate the agent
    deterministic: Whether to use deterministic or stochastic actions
    render: Whether to render the environment or not.
    video_caption: Wandb video caption.
    video_config: Video specifications.
    return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    """  
    episode_rewards = np.zeros(n_eval_episodes)
    episode_lengths = np.zeros(n_eval_episodes)

    for traffic_scene in eval_files:
        if verbose == 1:
            print(f"Evaluating policy on {traffic_scene}...")

        for eval_mode in eval_modes:
            # Reset env
            observations = env.reset(filename=traffic_scene)
            num_agents_controlled = len(env.agent_ids)
            curr_rewards = np.zeros(num_agents_controlled)
            frames = []
            
            for timestep in range(n_steps_per_episode):
                
                if eval_mode == 'policy':
                    # Predict actions
                    actions, _ = model.predict(
                        observations,
                        deterministic=deterministic,
                    )
                elif eval_mode == 'expert':
                    actions = None

                # Step environment
                new_observations, rewards, dones, infos = env.step(actions)

                for agent_idx, agent_id in enumerate(env.agent_ids):
                    if agent_id not in env.dead_agent_ids:
                        curr_rewards[agent_idx] += rewards[agent_idx]

                observations = new_observations

                # Render
                if render:
                    if timestep % video_config.logging.render_interval == 0:
                        if video_config.logging.where_am_i == "headless_machine":
                            with Display() as disp:
                                render_scene = env.env.scenario.getImage(**video_config.render)
                                frames.append(render_scene.T)
                        else:
                            render_scene = env.scenario.getImage(**video_config)
                            frames.append(render_scene.T)
                
                if sum(dones) == env.num_agents or timestep == (n_steps_per_episode-1):
                    episode_rewards = curr_rewards.sum() / env.num_agents
                    episode_lengths = timestep
                    break

            # Log video to wandb
            if render:
                movie_frames = np.array(frames, dtype=np.uint8)
                video_key = f"Scene (N = {env.num_agents}): {env.env.file}" 
                wandb.log(
                    {
                        video_key: wandb.Video(movie_frames, 
                        fps=video_config.logging.fps, 
                        caption=f'{video_caption}_mode_:{eval_mode}'),
                    },
                )

    # Close env
    env.close()

    if return_episode_rewards:
        return episode_rewards, episode_lengths
    else:
        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)    
        return mean_reward, std_reward

