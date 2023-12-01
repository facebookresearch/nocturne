# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Code modified from https://github.com/marlbenchmark/on-policy
"""Runner for PPO from https://github.com/marlbenchmark/on-policy."""
from pathlib import Path
import os
import time

import hydra
from cfgs.config import set_display_window
import imageio
import numpy as np
import setproctitle
import torch
import wandb

# from algos.ppo.base_runner import Runner
from algos.ppo.a3c_runner import Runner
from algos.ppo.env_wrappers import SubprocVecEnv, DummyVecEnv

from nocturne.envs.wrappers import create_ppo_env


def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()


def make_train_env(cfg):
    """Construct a training environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg, rank)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env

        return init_env

    if cfg.algorithm.n_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(cfg.algorithm.n_rollout_threads)])


def make_eval_env(cfg):
    """Construct an eval environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env

        return init_env

    if cfg.algorithm.n_eval_rollout_threads == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv(
            [get_env_fn(i) for i in range(cfg.algorithm.n_eval_rollout_threads)])


def make_render_env(cfg):
    """Construct a rendering environment."""

    def get_env_fn(rank):

        def init_env():
            env = create_ppo_env(cfg)
            # TODO(eugenevinitsky) implement this
            env.seed(cfg.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(0)])


class NocturneSharedRunner(Runner):
    """
    Runner class to perform training, evaluation and data collection for the Nocturne envs.

    WARNING: Assumes a shared policy.
    """

    def __init__(self, config):
        """Initialize."""
        super(NocturneSharedRunner, self).__init__(config)
        self.cfg = config['cfg.algo']
        self.render_envs = config['render_envs']

    def run(self):
        """Run the training code."""
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps
                       ) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(
                    step)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                    .format(self.algorithm_name, self.experiment_name,
                            episode * self.n_rollout_threads,
                            episodes * self.n_rollout_threads, total_num_steps,
                            self.num_env_steps,
                            int(total_num_steps / (end - start))))

                if self.use_wandb:
                    wandb.log({'fps': int(total_num_steps / (end - start))},
                              step=total_num_steps)
                env_infos = {}
                for agent_id in range(self.num_agents):
                    idv_rews = []
                    for info in infos:
                        if 'individual_reward' in info[agent_id].keys():
                            idv_rews.append(
                                info[agent_id]['individual_reward'])
                    agent_k = 'agent%i/individual_rewards' % agent_id
                    env_infos[agent_k] = idv_rews

                # TODO(eugenevinitsky) this does not correctly account for the fact that there could be
                # two episodes in the buffer
                train_infos["average_episode_rewards"] = np.mean(
                    self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(
                    train_infos["average_episode_rewards"]))
                print(
                    f"maximum per step reward is {np.max(self.buffer.rewards)}"
                )
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

            # save videos
            if episode % self.cfg.render_interval == 0:
                self.render(total_num_steps)

    def warmup(self):
        """Initialize the buffers."""
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents,
                                                            axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        """Collect rollout data."""
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads))
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] +
                                        1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env),
                                                 axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(
                np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        """Store the data in the buffers."""
        obs, rewards, dones, _, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env] = np.zeros(((dones_env).sum(), self.num_agents,
                                          self.recurrent_N, self.hidden_size),
                                         dtype=np.float32)
        rnn_states_critic[dones_env] = np.zeros(
            ((dones_env).sum(), self.num_agents,
             *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                        dtype=np.float32)
        masks[dones_env] = np.zeros(((dones_env).sum(), self.num_agents, 1),
                                    dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1),
                               dtype=np.float32)
        active_masks[dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)
        active_masks[dones_env] = np.ones(
            ((dones_env).sum(), self.num_agents, 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents,
                                                            axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs,
                           obs,
                           rnn_states,
                           rnn_states_critic,
                           actions,
                           action_log_probs,
                           values,
                           rewards,
                           masks,
                           active_masks=active_masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        """Get the policy returns in deterministic mode."""
        eval_episode = 0

        eval_episode_rewards = []
        one_episode_rewards = [[] for _ in range(self.n_eval_rollout_threads)]
        num_achieved_goals = 0
        num_collisions = 0

        i = 0
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N,
             self.hidden_size),
            dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1),
                             dtype=np.float32)

        while eval_episode < self.cfg.eval_episodes:
            i += 1
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = \
                self.trainer.policy.act(np.concatenate(eval_obs),
                                        np.concatenate(eval_rnn_states),
                                        np.concatenate(eval_masks),
                                        deterministic=True)
            eval_actions = np.array(
                np.split(_t2n(eval_actions), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            # Observed reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(
                eval_actions)
            for info_arr in eval_infos:
                for agent_info_arr in info_arr:
                    if 'goal_achieved' in agent_info_arr and agent_info_arr[
                            'goal_achieved']:
                        num_achieved_goals += 1
                    if 'collided' in agent_info_arr and agent_info_arr[
                            'collided']:
                        num_collisions += 1

            for i in range(self.n_eval_rollout_threads):
                one_episode_rewards[i].append(eval_rewards[i])

            eval_dones_env = np.all(eval_dones, axis=1)

            eval_rnn_states[eval_dones_env] = np.zeros(
                ((eval_dones_env).sum(), self.num_agents, self.recurrent_N,
                 self.hidden_size),
                dtype=np.float32)

            eval_masks = np.ones(
                (self.n_eval_rollout_threads, self.num_agents, 1),
                dtype=np.float32)
            eval_masks[eval_dones_env] = np.zeros(
                ((eval_dones_env).sum(), self.num_agents, 1), dtype=np.float32)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(
                        np.sum(one_episode_rewards[eval_i], axis=0).mean())
                    one_episode_rewards[eval_i] = []

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_episode_rewards = np.mean(eval_episode_rewards)
        if self.use_wandb:
            wandb.log({'eval_episode_rewards': eval_episode_rewards},
                      step=total_num_steps)
            wandb.log(
                {
                    'avg_eval_goals_achieved':
                    num_achieved_goals / self.num_agents /
                    self.cfg.eval_episodes
                },
                step=total_num_steps)
            wandb.log(
                {
                    'avg_eval_num_collisions':
                    num_collisions / self.num_agents / self.cfg.eval_episodes
                },
                step=total_num_steps)

    @torch.no_grad()
    def render(self, total_num_steps):
        """Visualize the env."""
        envs = self.render_envs

        all_frames = []
        for episode in range(self.cfg.render_episodes):
            obs = envs.reset()
            if self.cfg.save_gifs:
                image = envs.envs[0].render('rgb_array')
                all_frames.append(image)
            else:
                envs.render('human')

            rnn_states = np.zeros(
                (1, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            masks = np.ones((1, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            self.trainer.prep_rollout()
            for step in range(self.episode_length):
                calc_start = time.time()

                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True)
                actions = np.array(np.split(_t2n(action), 1))
                rnn_states = np.array(np.split(_t2n(rnn_states), 1))

                if envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
                    for i in range(envs.action_space[0].shape):
                        uc_actions_env = np.eye(envs.action_space[0].high[i] +
                                                1)[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate(
                                (actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == 'Discrete':
                    actions_env = np.squeeze(
                        np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones] = np.zeros(
                    ((dones).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32)
                masks = np.ones((1, self.num_agents, 1), dtype=np.float32)
                masks[dones] = np.zeros(((dones).sum(), 1), dtype=np.float32)

                if self.cfg.save_gifs:
                    image = envs.envs[0].render('rgb_array')
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.cfg.ifi:
                        time.sleep(self.cfg.ifi - elapsed)
                else:
                    envs.render('human')

                if np.all(dones[0]):
                    break

            # note, every rendered episode is exactly the same since there's no randomness in the env and our actions
            # are deterministic
            # TODO(eugenevinitsky) why is this lower than the non-render reward?
            render_val = np.mean(np.sum(np.array(episode_rewards), axis=0))
            print("episode reward of rendered episode is: " + str(render_val))
            if self.use_wandb:
                wandb.log({'render_rew': render_val}, step=total_num_steps)

        if self.cfg.save_gifs:
            if self.use_wandb:
                np_arr = np.stack(all_frames).transpose((0, 3, 1, 2))
                wandb.log({"video": wandb.Video(np_arr, fps=4, format="gif")},
                          step=total_num_steps)
            # else:
            imageio.mimsave(os.getcwd() + '/render.gif',
                            all_frames,
                            duration=self.cfg.ifi)


@hydra.main(config_path='../../cfgs/', config_name='config')
def main(cfg):
    """Run the on-policy code."""
    set_display_window()
    logdir = Path(os.getcwd())
    if cfg.wandb_id is not None:
        wandb_id = cfg.wandb_id
    else:
        wandb_id = wandb.util.generate_id()
        # with open(os.path.join(logdir, 'wandb_id.txt'), 'w+') as f:
        #     f.write(wandb_id)
    wandb_mode = "disabled" if (cfg.debug or not cfg.wandb) else "online"

    if cfg.wandb:
        run = wandb.init(config=cfg,
                         project=cfg.wandb_name,
                         name=wandb_id,
                         group='ppov2_' + cfg.experiment,
                         resume="allow",
                         settings=wandb.Settings(start_method="fork"),
                         mode=wandb_mode)
    else:
        if not logdir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [
                int(str(folder.name).split('run')[1])
                for folder in logdir.iterdir()
                if str(folder.name).startswith('run')
            ]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        logdir = logdir / curr_run
        if not logdir.exists():
            os.makedirs(str(logdir))

    if cfg.algorithm.algorithm_name == "rmappo":
        assert (cfg.algorithm.use_recurrent_policy
                or cfg.algorithm.use_naive_recurrent_policy), (
                    "check recurrent policy!")
    elif cfg.algorithm.algorithm_name == "mappo":
        assert (not cfg.algorithm.use_recurrent_policy
                and not cfg.algorithm.use_naive_recurrent_policy), (
                    "check recurrent policy!")
    else:
        raise NotImplementedError

    # cuda
    if 'cpu' not in cfg.algorithm.device and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device(cfg.algorithm.device)
        torch.set_num_threads(cfg.algorithm.n_training_threads)
        # if cfg.algorithm.cuda_deterministic:
        #     import torch.backends.cudnn as cudnn
        #     cudnn.benchmark = False
        #     cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(cfg.algorithm.n_training_threads)

    setproctitle.setproctitle(
        str(cfg.algorithm.algorithm_name) + "-" + str(cfg.experiment))

    # seed
    torch.manual_seed(cfg.algorithm.seed)
    torch.cuda.manual_seed_all(cfg.algorithm.seed)
    np.random.seed(cfg.algorithm.seed)

    # env init
    # TODO(eugenevinitsky) this code requires a fixed number of agents but this
    # should be done by overriding in the hydra config rather than here
    cfg.subscriber.keep_inactive_agents = True
    envs = make_train_env(cfg)
    eval_envs = make_eval_env(cfg)
    render_envs = make_render_env(cfg)
    # TODO(eugenevinitsky) hacky
    num_agents = envs.reset().shape[1]

    config = {
        "cfg.algo": cfg.algorithm,
        "envs": envs,
        "eval_envs": eval_envs,
        "render_envs": render_envs,
        "num_agents": num_agents,
        "device": device,
        "logdir": logdir
    }

    # run experiments
    runner = NocturneSharedRunner(config)
    runner.run()

    # post process
    envs.close()
    if cfg.algorithm.use_eval and eval_envs is not envs:
        eval_envs.close()

    if cfg.wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(
            str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == '__main__':
    main()
