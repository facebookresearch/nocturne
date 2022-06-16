# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Use to create movies of trained policies."""
import argparse
from collections import deque
import json
import sys
import time
import os

import imageio
import matplotlib.pyplot as plt
import numpy as np
from pyvirtualdisplay import Display
import torch

from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution, \
     CategoricalActionDistribution
from sample_factory.algorithms.utils.arguments import load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict

from run_sample_factory import register_custom_components

from cfgs.config import PROCESSED_TRAIN_NO_TL, PROCESSED_VALID_NO_TL, PROJECT_PATH  # noqa: F401


def run_eval(cfg_dict, max_num_frames=1e9):
    """Run evaluation over a single file. Exits when one episode finishes.

    Args:
        cfg (dict): configuration file for instantiating the agents and environment.
        max_num_frames (int, optional): Deprecated. Should be removed.

    Returns
    -------
        None: None

    """
    cfg = load_from_checkpoint(cfg_dict)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1
    cfg.seed = np.random.randint(10000)
    cfg.scenario_path = cfg_dict.scenario_path

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))

    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space,
                                       env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(
        LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    num_frames = 0

    last_render_start = time.time()

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    obs = env.reset()
    print(os.path.join(env.cfg['scenario_path'], env.unwrapped.file))
    rnn_states = torch.zeros(
        [env.num_agents, get_hidden_size(cfg)],
        dtype=torch.float32,
        device=device)
    episode_reward = np.zeros(env.num_agents)
    finished_episode = [False] * env.num_agents

    if not cfg.no_render:
        fig = plt.figure()
        frames = []
        ego_frames = []
        feature_frames = []

    with torch.no_grad():
        while not max_frames_reached(num_frames):
            obs_torch = AttrDict(transform_dict_observations(obs))
            for key, x in obs_torch.items():
                obs_torch[key] = torch.from_numpy(x).to(device).float()

            policy_outputs = actor_critic(obs_torch,
                                          rnn_states,
                                          with_action_distribution=True)

            # sample actions from the distribution by default
            actions = policy_outputs.actions

            action_distribution = policy_outputs.action_distribution
            if isinstance(action_distribution, ContinuousActionDistribution):
                if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                    actions = action_distribution.means
            if isinstance(action_distribution, CategoricalActionDistribution):
                if not cfg.discrete_actions_sample:
                    actions = policy_outputs['action_logits'].argmax(axis=1)

            actions = actions.cpu().numpy()

            rnn_states = policy_outputs.rnn_states

            for _ in range(render_action_repeat):
                if not cfg.no_render:
                    target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
                    current_delay = time.time() - last_render_start
                    time_wait = target_delay - current_delay

                    if time_wait > 0:
                        # log.info('Wait time %.3f', time_wait)
                        time.sleep(time_wait)

                    last_render_start = time.time()
                    img = env.render()
                    frames.append(img)
                    ego_img = env.render_ego()
                    if ego_img is not None:
                        ego_frames.append(ego_img)
                    feature_img = env.render_features()
                    if feature_img is not None:
                        feature_frames.append(feature_img)

                obs, rew, done, infos = env.step(actions)

                episode_reward += rew
                num_frames += 1

                for agent_i, done_flag in enumerate(done):
                    if done_flag:
                        finished_episode[agent_i] = True
                        episode_rewards[agent_i].append(
                            episode_reward[agent_i])
                        true_rewards[agent_i].append(infos[agent_i].get(
                            'true_reward', episode_reward[agent_i]))
                        log.info(
                            'Episode finished for agent %d at %d frames. Reward: %.3f, true_reward: %.3f',
                            agent_i, num_frames, episode_reward[agent_i],
                            true_rewards[agent_i][-1])
                        rnn_states[agent_i] = torch.zeros(
                            [get_hidden_size(cfg)],
                            dtype=torch.float32,
                            device=device)
                        episode_reward[agent_i] = 0

                # if episode terminated synchronously for all agents, pause a bit before starting a new one
                if all(done):
                    if not cfg.no_render:
                        imageio.mimsave(os.path.join(PROJECT_PATH,
                                                     'animation.mp4'),
                                        np.array(frames),
                                        fps=30)
                        plt.close(fig)
                        imageio.mimsave(os.path.join(PROJECT_PATH,
                                                     'animation_ego.mp4'),
                                        np.array(ego_frames),
                                        fps=30)
                        plt.close(fig)
                        imageio.mimsave(os.path.join(PROJECT_PATH,
                                                     'animation_feature.mp4'),
                                        np.array(feature_frames),
                                        fps=30)
                        plt.close(fig)
                    if not cfg.no_render:
                        env.render()
                    time.sleep(0.05)

                if all(finished_episode):
                    finished_episode = [False] * env.num_agents
                    avg_episode_rewards_str, avg_true_reward_str = '', ''
                    for agent_i in range(env.num_agents):
                        avg_rew = np.mean(episode_rewards[agent_i])
                        avg_true_rew = np.mean(true_rewards[agent_i])
                        if not np.isnan(avg_rew):
                            if avg_episode_rewards_str:
                                avg_episode_rewards_str += ', '
                            avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
                        if not np.isnan(avg_true_rew):
                            if avg_true_reward_str:
                                avg_true_reward_str += ', '
                            avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'
                    avg_goal = infos[0]['episode_extra_stats']['goal_achieved']
                    avg_collisions = infos[0]['episode_extra_stats'][
                        'collided']
                    log.info(f'Avg goal achieved, {avg_goal}')
                    log.info(f'Avg num collisions, {avg_collisions}')
                    log.info('Avg episode rewards: %s, true rewards: %s',
                             avg_episode_rewards_str, avg_true_reward_str)
                    log.info(
                        'Avg episode reward: %.3f, avg true_reward: %.3f',
                        np.mean([
                            np.mean(episode_rewards[i])
                            for i in range(env.num_agents)
                        ]),
                        np.mean([
                            np.mean(true_rewards[i])
                            for i in range(env.num_agents)
                        ]))
                    return avg_goal
    env.close()


def main():
    """Script entry point."""
    disp = Display()
    disp.start()
    register_custom_components()

    parser = argparse.ArgumentParser()
    parser.add_argument('cfg_path', type=str)
    args = parser.parse_args()

    file_path = os.path.join(args.cfg_path, 'cfg.json')
    with open(file_path, 'r') as file:
        cfg_dict = json.load(file)

    cfg_dict['cli_args'] = {}
    cfg_dict['fps'] = 0
    cfg_dict['render_action_repeat'] = None
    cfg_dict['no_render'] = False
    cfg_dict['policy_index'] = 0
    cfg_dict['record_to'] = os.path.join(os.getcwd(), '..', 'recs')
    cfg_dict['continuous_actions_sample'] = True
    cfg_dict['discrete_actions_sample'] = False
    cfg_dict['remove_at_collide'] = True
    cfg_dict['remove_at_goal'] = True
    cfg_dict['scenario_path'] = PROCESSED_VALID_NO_TL

    class Bunch(object):

        def __init__(self, adict):
            self.__dict__.update(adict)

    cfg = Bunch(cfg_dict)
    avg_goals = []
    for _ in range(1):
        avg_goal = run_eval(cfg)
        avg_goals.append(avg_goal)
    print(avg_goals)
    print('the total average goal achieved is {}'.format(np.mean(avg_goals)))


if __name__ == '__main__':
    sys.exit(main())
