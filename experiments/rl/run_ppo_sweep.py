"""Cast a multi-agent env as vec env to use SB3's PPO."""
import logging
from datetime import datetime
import torch
import wandb

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from utils.render import make_video

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.ma_ppo import MultiAgentPPO
from configs.sweep_configs import sweep_config_arch
from utils.string_utils import datetime_to_str

logging.basicConfig(level=logging.INFO)

def train_func():
    
    # Load environment and experiment configurations
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    video_config = load_config("video_config")

    # Set up run
    date_time = datetime_to_str(dt=datetime.now())
    RUN_ID = f"{exp_config.exp_name}_{date_time}"
    run = wandb.init(
        project=exp_config.project,
        name=RUN_ID,
        id=RUN_ID,
        **exp_config.wandb,
    )

    # GET PARAMETERS
    TOTAL_TIMESTEPS = wandb.config.total_timesteps
    SEED = wandb.config.seed
    ENT_COEF = wandb.config.ent_coef
    VF_COEF = wandb.config.vf_coef
    ACT_FUNC = torch.nn.ReLU if wandb.config.policy_act_func == "relu" else torch.nn.Tanh
    policy_kwargs = dict( # Use custom MLP policy 
        activation_fn=ACT_FUNC, 
        net_arch=wandb.config.policy_layers
    )

    # Set the maximum number of agents to control
    env_config.max_num_vehicles = wandb.config.num_controlled_agents
    env_config.n_frames_stacked = wandb.config.memory

    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config, 
        num_envs=env_config.max_num_vehicles
    )
     logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")
    logging.info(f"Learning in {env_config.num_files} scene(s): {env.env.files}")
    logging.info(f"--- obs_space: {env.observation_space.shape[0]} ---")
    logging.info(f"Action_space\n: {env.env.idx_to_actions}")

    # Set device
    exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize custom callback
    custom_callback = CustomMultiAgentCallback(
        env_config=env_config,
        exp_config=exp_config,
        video_config=video_config,
        wandb_run=run,
    )
 
    # Make scene init video to check expert actions
    if exp_config.track_wandb:
        for model in exp_config.wandb_init_videos:
            make_video(
                env_config=env_config,
                exp_config=exp_config,
                video_config=video_config,
                model=model,
                n_steps=None,
            )

    # Set up PPO model
    model = MultiAgentPPO(
        n_steps=exp_config.ppo.n_steps,
        policy=exp_config.ppo.policy,
        policy_kwargs=policy_kwargs,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        env=env,
        seed=SEED, # Seed for the pseudo random generators
        tensorboard_log=f"runs/{RUN_ID}",
        verbose=1,
        device=exp_config.ppo.device,
    )

    # Learn
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=custom_callback,
    )

    # Finish
    if exp_config.track_wandb:
        run.finish()

if __name__ == "__main__":

    # Create sweep id
    sweep_id = wandb.sweep(sweep_config_arch)

    # Run sweep
    wandb.agent(sweep_id, function=train_func)