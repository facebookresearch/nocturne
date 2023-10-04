
import gymnasium
import numpy as np
import gym

class NocturneToSB3(gymnasium.Env):
    """Makes Nocturne env compatible with SB3.
    ! NOTE: Controlling a single agent.
    """

    def __init__(self, nocturne_env: gym.Env):
        self.env = nocturne_env
        self.action_space = gymnasium.spaces.Discrete(self.env.action_space.n)
        self.observation_space = gymnasium.spaces.Box(
            -np.inf, np.inf, self.env.observation_space.shape, np.float32
        )
    
    def step(self, action):
        """Take a step in the environment, convert dicts to np arrays.

        Args:
            action (Dict): Dictionary with a single action for the controlled vehicle.

        Returns:
            observation, reward, terminated, truncated, info (np.ndarray, float, bool, bool, dict)
        """
        next_obs_dict, rewards_dict, dones_dict, info_dict = self.env.step(
            action_dict={self.controlled_vehicle: action}
        )

        return (
            next_obs_dict[self.controlled_vehicle],
            rewards_dict[self.controlled_vehicle],
            dones_dict[self.controlled_vehicle],
            False,
            info_dict[self.controlled_vehicle],
        )
    
    def reset(self, seed=None):
        """Reset the environment."""
        obs_dict = self.env.reset()
        assert (
            len(self.env.controlled_vehicles) == 1
        ), "This wrapper does not support multi-agent control."
        
        self.controlled_vehicle = self.env.controlled_vehicles[0].id
        return obs_dict[self.controlled_vehicle], {}

    @property
    def action_space(self):
        return self.env.action_space

    @action_space.setter
    def action_space(self, action_space):
        self.env.action_space = action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self.env.observation_space = observation_space
    
    def render(self):
        pass

    def close(self):
        pass

    @property
    def seed(self, seed=None):
        return None

    @seed.setter
    def seed(self, seed=None):
        pass

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_attr(self, attr_name: str):
        return getattr(self._env, attr_name)

    def set_attr(self, attr_name: str):
        setattr(self._env, attr_name)