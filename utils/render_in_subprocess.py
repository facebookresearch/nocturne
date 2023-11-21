from pyvirtualdisplay import Display
from pathlib import Path
import numpy as np
import torch
import typer
import pickle
from nocturne.envs.base_env import BaseEnv
from utils.render import discretize_action

def render(
    directory: str,
    model: str,
    filename: str,
    scene_idx: int,
    max_steps: int,
    snap_interval: int,
    deterministic: bool = True,
    headless_machine: bool = True,
):
    # Ensure directory is a Path object
    directory = Path(directory)

    # Load env config, video config, and model
    with open(directory / "env_config.pkl", "rb") as env_config_file:
        env_config = pickle.load(env_config_file)
    with open(directory / "video_config.pkl", "rb") as video_config_file:
        video_config = pickle.load(video_config_file)
    if model == "custom":
        with open(directory / "policy_model.pkl", "rb") as model_file:
            model = pickle.load(model_file)

    # Make env
    env = BaseEnv(env_config)

    # Select correct scene
    for scene_idx in range(scene_idx + 1):
        if "tfrecord" in filename:
            next_obs_dict = env.reset(filename)
        else:
            next_obs_dict = env.reset()

        next_done_dict = {agent_id: False for agent_id in next_obs_dict}

    # Render the frames
    frames = []
    for timestep in range(max_steps):
        action_dict = {}
        for agent in env.controlled_vehicles:
            if agent.id in next_obs_dict and not next_done_dict[agent.id]:
                if model in ("expert", "expert_discrete"):
                    agent.expert_control = True
                    action = env.scenario.expert_action(agent, timestep)
                    agent.expert_control = False
                    if model == "expert_discrete":
                        if action is not None:
                            action, action_idx = discretize_action(env_config=env_config, action=action)
                    action_dict[agent.id] = action
                    if model == "expert":
                        agent.expert_control = True
                        action_dict = {}
                else:
                    obs_tensor = torch.Tensor(next_obs_dict[agent.id]).unsqueeze(dim=0)
                    with torch.no_grad():
                        action_idx, _ = model.predict(obs_tensor, deterministic=deterministic)
                    action_dict[agent.id] = action_idx.item()

        next_obs_dict, _, next_done_dict, _ = env.step(action_dict)

        if model in ("expert", "expert_discrete"):
            action_dict = {
                agent_id: discretize_action(env_config, action)[1] for agent_id, agent in action_dict.items() if action is not None
            }

        if timestep % snap_interval == 0:
            # If we're on a headless machine: activate display and render
            if headless_machine:
                with Display(backend="xvfb") as disp:
                    render_scene = env.scenario.getImage(**video_config.render)
                    frames.append(render_scene.T)
            else:
                render_scene = env.scenario.getImage(**video_config.render)
                frames.append(render_scene.T)

        if next_done_dict["__all__"]:
            break
    
    # Convert frames to numpy array
    movie_frames = np.array(frames, dtype=np.uint8)

    # Write movie_frames to pickle
    with open(directory / "frames.pkl", "wb") as movie_frames_file:
        pickle.dump(obj=movie_frames, file=movie_frames_file)

if __name__ == "__main__":
    typer.run(render)