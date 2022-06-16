# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Runner script for sample factory.

To run in single agent mode on one file for testing.
python -m run_sample_factory algorithm=APPO ++algorithm.train_in_background_thread=True \
    ++algorithm.num_workers=10 ++algorithm.experiment=EXPERIMENT_NAME \
    ++max_num_vehicles=1 ++num_files=1

To run in multiagent mode on one file for testing
python -m run_sample_factory algorithm=APPO ++algorithm.train_in_background_thread=True \
    ++algorithm.num_workers=10 ++algorithm.experiment=EXPERIMENT_NAME \
    ++num_files=1

To run on all files set ++num_files=-1

For debugging
python -m run_sample_factory algorithm=APPO ++algorithm.train_in_background_thread=False \
    ++algorithm.num_workers=1 ++force_envs_single_thread=False
After training for a desired period of time, evaluate the policy by running:
python -m sample_factory_examples.enjoy_custom_multi_env --algo=APPO \
    --env=my_custom_multi_env_v1 --experiment=example
"""
import os
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
from sample_factory.envs.env_registry import global_env_registry
from sample_factory.run_algorithm import run_algorithm
from sample_factory_examples.train_custom_env_custom_model import override_default_params_func
from sample_factory.algorithms.appo.model_utils import get_obs_shape, EncoderBase, nonlinearity, register_custom_encoder
from torch import nn

from nocturne.envs.wrappers import create_env


class SampleFactoryEnv():
    """Wrapper environment that converts between our dicts and Sample Factory format."""

    def __init__(self, env):
        """Initialize wrapper.

        Args
        ----
            env (BaseEnv): Base environment that we are wrapping.
        """
        self.env = env
        self.num_agents = self.env.cfg['max_num_vehicles']
        self.agent_ids = [i for i in range(self.num_agents)]
        self.is_multiagent = True
        _ = self.env.reset()
        # used to track which agents are done
        self.already_done = [False for _ in self.agent_ids]
        self.episode_rewards = np.zeros(self.num_agents)

    def step(self, actions):
        """Convert between environment dicts and sample factory lists.

        Important to note:
        1) Items in info['episode_extra_stats'] will be logged by sample factory.
        2) sample factory does not reset the environment for you
           so we reset it if the env returns __all__ in its done dict

        Args:
            actions ({str: numpy array}): agent actions

        Returns
        -------
            obs_n ([np.array]): N length list of agent observations
            rew_n ([float]): N length list of agent rewards
            info_n ([{str: float}]): N length list of info dicts
            done_n ([bool]): N length list of whether agents are done

        """
        agent_actions = {}
        for action, agent_id, already_done in zip(actions, self.agent_ids,
                                                  self.already_done):
            if already_done:
                continue
            agent_actions[self.agent_id_to_env_id_map[agent_id]] = action
        next_obses, rew, done, info = self.env.step(agent_actions)
        rew_n = []
        done_n = []
        info_n = []

        for agent_id in self.agent_ids:
            # first check that the agent_id ever had a corresponding vehicle
            # and then check that there's actually an observation for it i.e. it's not done
            if agent_id in self.agent_id_to_env_id_map.keys(
            ) and self.agent_id_to_env_id_map[agent_id] in next_obses.keys():
                map_key = self.agent_id_to_env_id_map[agent_id]
                # since the environment may have just reset, we don't actually have
                # reward objects yet
                rew_n.append(rew.get(map_key, 0))
                agent_info = info.get(map_key, {})
                # track the per-agent reward for later logging
                self.episode_rewards[agent_id] += rew.get(map_key, 0)
                self.num_steps[agent_id] += 1
                self.goal_achieved[agent_id] = self.goal_achieved[
                    agent_id] or agent_info['goal_achieved']
                self.collided[agent_id] = self.collided[
                    agent_id] or agent_info['collided']
                self.veh_edge_collided[agent_id] = self.veh_edge_collided[
                    agent_id] or agent_info['veh_edge_collision']
                self.veh_veh_collided[agent_id] = self.veh_veh_collided[
                    agent_id] or agent_info['veh_veh_collision']
            else:
                rew_n.append(0)
                agent_info = {}
            if self.already_done[agent_id]:
                agent_info['is_active'] = False
            else:
                agent_info['is_active'] = True
            info_n.append(agent_info)
        # now stick in some extra state information if needed
        # anything in episode_extra_stats is logged at the end of the episode
        if done['__all__']:
            # log any extra info that you need
            avg_rew = np.mean(self.episode_rewards[self.valid_indices])
            avg_len = np.mean(self.num_steps[self.valid_indices])
            avg_goal_achieved = np.mean(self.goal_achieved[self.valid_indices])
            avg_collided = np.mean(self.collided[self.valid_indices])
            avg_veh_edge_collided = np.mean(
                self.veh_edge_collided[self.valid_indices])
            avg_veh_veh_collided = np.mean(
                self.veh_veh_collided[self.valid_indices])
            for info in info_n:
                info['episode_extra_stats'] = {}
                info['episode_extra_stats']['avg_rew'] = avg_rew
                info['episode_extra_stats']['avg_agent_len'] = avg_len
                info['episode_extra_stats'][
                    'goal_achieved'] = avg_goal_achieved
                info['episode_extra_stats']['collided'] = avg_collided
                info['episode_extra_stats'][
                    'veh_edge_collision'] = avg_veh_edge_collided
                info['episode_extra_stats'][
                    'veh_veh_collision'] = avg_veh_veh_collided

        # update the dones so we know if we need to reset
        # sample factory does not call reset for you
        for env_id, done_val in done.items():
            # handle the __all__ signal that's just in there for
            # telling when the environment should stop
            if env_id == '__all__':
                continue
            if done_val:
                agent_id = self.env_id_to_agent_id_map[env_id]
                self.already_done[agent_id] = True

        # okay, now if all the agents are done set done to True for all of them
        # otherwise, False. Sample factory uses info['is_active'] to track if agents
        # are done, not the done signal
        # also, convert the obs_dict into the right format
        if done['__all__']:
            done_n = [True] * self.num_agents
            obs_n = self.reset()
        else:
            done_n = [False] * self.num_agents
            obs_n = self.obs_dict_to_list(next_obses)
        return obs_n, rew_n, done_n, info_n

    def obs_dict_to_list(self, obs_dict):
        """Convert the dictionary returned by the environment into a fixed size list of arrays.

        Args:
            obs_dict ({agent id in environment: observation}): dict mapping ID to observation

        Returns
        -------
            [np.array]: List of arrays ordered by which agent ID they correspond to.
        """
        obs_n = []
        for agent_id in self.agent_ids:
            # first check that the agent_id ever had a corresponding vehicle
            # and then check that there's actually an observation for it i.e. it's not done
            if agent_id in self.agent_id_to_env_id_map.keys(
            ) and self.agent_id_to_env_id_map[agent_id] in obs_dict.keys():
                map_key = self.agent_id_to_env_id_map[agent_id]
                obs_n.append(obs_dict[map_key])
            else:
                obs_n.append(self.dead_feat)
        return obs_n

    def reset(self):
        """Reset the environment.

        Key things done here:
        1) build a map between the agent IDs in the environment (which are not necessarily 0-N)
           and the agent IDs for sample factory which are from 0 to the maximum number of agents
        2) sample factory (until some bugs are fixed) requires a fixed number of agents. Some of these
           agents will be dummy agents that do not act in the environment. So, here we build valid
           indices which can be used to figure out which agent IDs correspond

        Returns
        -------
            [np.array]: List of numpy arrays, one for each agent.
        """
        # track the agent_ids that actually take an action during the episode
        self.valid_indices = []
        self.episode_rewards = np.zeros(self.num_agents)
        self.num_steps = np.zeros(self.num_agents)
        self.goal_achieved = np.zeros(self.num_agents)
        self.collided = np.zeros(self.num_agents)
        self.veh_veh_collided = np.zeros(self.num_agents)
        self.veh_edge_collided = np.zeros(self.num_agents)
        self.already_done = [False for _ in self.agent_ids]
        next_obses = self.env.reset()
        env_keys = sorted(list(next_obses.keys()))
        # agent ids is a list going from 0 to (num_agents - 1)
        # however, the vehicle IDs might go from 0 to anything
        # we want to initialize a mapping that is maintained through the episode and always
        # uniquely convert the vehicle ID to an agent id
        self.agent_id_to_env_id_map = {
            agent_id: env_id
            for agent_id, env_id in zip(self.agent_ids, env_keys)
        }
        self.env_id_to_agent_id_map = {
            env_id: agent_id
            for agent_id, env_id in zip(self.agent_ids, env_keys)
        }
        # if there isn't a mapping from an agent id to a vehicle id, that agent should be
        # set to permanently inactive
        for agent_id in self.agent_ids:
            if agent_id not in self.agent_id_to_env_id_map.keys():
                self.already_done[agent_id] = True
            else:
                # check that this isn't actually a fake padding agent used
                # when keep_inactive_agents is True
                if agent_id in self.agent_id_to_env_id_map.keys(
                ) and self.agent_id_to_env_id_map[
                        agent_id] not in self.env.dead_agent_ids:
                    self.valid_indices.append(agent_id)
        obs_n = self.obs_dict_to_list(next_obses)
        return obs_n

    @property
    def observation_space(self):
        """See superclass."""
        return self.env.observation_space

    @property
    def action_space(self):
        """See superclass."""
        return self.env.action_space

    def render(self, mode=None):
        """See superclass."""
        return self.env.render(mode)

    def seed(self, seed=None):
        """Pass the seed to the environment."""
        self.env.seed(seed)

    def __getattr__(self, name):
        """Pass attributes directly through to the wrapped env. TODO(remove)."""
        return getattr(self.env, name)


class CustomEncoder(EncoderBase):
    """Encoder for the input."""

    def __init__(self, cfg, obs_space, timing):
        super().__init__(cfg, timing)

        obs_shape = get_obs_shape(obs_space)
        assert len(obs_shape.obs) == 1

        fc_encoder_layer = cfg.encoder_hidden_size
        encoder_layers = [
            nn.Linear(obs_shape.obs[0], fc_encoder_layer),
            nonlinearity(cfg),
            nn.Linear(fc_encoder_layer, fc_encoder_layer),
            nonlinearity(cfg),
        ]

        self.mlp_head = nn.Sequential(*encoder_layers)
        self.init_fc_blocks(fc_encoder_layer)

    def forward(self, obs_dict):
        """See superclass."""
        x = self.mlp_head(obs_dict['obs'])
        x = self.forward_fc_blocks(x)
        return x


def make_custom_multi_env_func(full_env_name, cfg, env_config=None):
    """Return a wrapped base environment.

    Args:
        full_env_name (str): Unused.
        cfg (dict): Dict needed to configure the environment.
        env_config (dict, optional): Deprecated. Will be removed from SampleFactory later.

    Returns
    -------
        SampleFactoryEnv: Wrapped environment.
    """
    env = create_env(cfg)
    return SampleFactoryEnv(env)


def register_custom_components():
    """Register needed constructors for custom environments."""
    global_env_registry().register_env(
        env_name_prefix='my_custom_multi_env_',
        make_env_func=make_custom_multi_env_func,
        override_default_params_func=override_default_params_func,
    )
    register_custom_encoder('custom_env_encoder', CustomEncoder)


@hydra.main(config_path="../../cfgs/", config_name="config")
def main(cfg):
    """Script entry point."""
    register_custom_components()
    # cfg = parse_args()
    # TODO(ev) hacky renaming and restructuring, better to do this cleanly
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    # copy algo keys into the main keys
    for key, value in cfg_dict['algorithm'].items():
        cfg_dict[key] = value
    # we didn't set a train directory so use the hydra one
    if cfg_dict['train_dir'] is None:
        cfg_dict['train_dir'] = os.getcwd()
        print(f'storing the results in {os.getcwd()}')
    else:
        output_dir = cfg_dict['train_dir']
        print(f'storing results in {output_dir}')

    # recommendation from Aleksei to keep horizon length fixed
    # and number of agents fixed and just pad missing / exited
    # agents with a vector of -1s
    cfg_dict['subscriber']['keep_inactive_agents'] = True

    # put it into a namespace so sample factory code runs correctly
    class Bunch(object):

        def __init__(self, adict):
            self.__dict__.update(adict)

    cfg = Bunch(cfg_dict)
    status = run_algorithm(cfg)
    return status


if __name__ == '__main__':
    sys.exit(main())
