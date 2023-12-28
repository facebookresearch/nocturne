import numpy as np
from box import Box

def discretize_action(env_config, action):
    """Discretize actions."""
    acceleration_actions = np.linspace(
        start=env_config.accel_lower_bound,
        stop=env_config.accel_upper_bound,
        num=env_config.accel_discretization,
    )
    acceleration_idx = np.abs(action.acceleration - acceleration_actions).argmin()
    action.acceleration = acceleration_actions[acceleration_idx]

    steering_actions = np.linspace(
        start=env_config.steering_lower_bound,
        stop=env_config.steering_upper_bound,
        num=env_config.steering_discretization,
    )
    steering_idx = np.abs(action.steering - steering_actions).argmin()
    action.steering = steering_actions[steering_idx]

    action_idx = acceleration_idx * env_config.steering_discretization + steering_idx

    return action, action_idx