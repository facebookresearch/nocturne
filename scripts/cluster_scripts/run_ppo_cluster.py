# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run on-policy PPO experiments on a SLURM cluster."""
import argparse
import os
import pathlib
import shutil
from datetime import datetime
from subprocess import Popen

from cfgs.config import PROJECT_PATH
from scripts.cluster_scripts.utils import Overrides


def make_code_snap(experiment, code_path, slurm_dir='exp'):
    """Copy code to directory to ensure that the run launches with correct commit.

    Args:
        experiment (str): Name of experiment
        code_path (str): Path to where we are saving the code.
        str_time (str): Unique time identifier used to distinguish
                        experiments with same name.

    Returns
    -------
        snap_dir (str): path to where the code has been copied.
    """
    now = datetime.now()
    if len(code_path) > 0:
        snap_dir = pathlib.Path(code_path) / slurm_dir
    else:
        snap_dir = pathlib.Path.cwd() / slurm_dir
    snap_dir /= now.strftime('%Y.%m.%d')
    snap_dir /= now.strftime('%H%M%S') + f'_{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(dir, pat):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            shutil.copy(f, dst_dir / f.name)

    dirs_to_copy = [
        '.', './cfgs/', './cfgs/algo', './algos/', './algos/ppo/',
        './algos/ppo/ppo_utils', './algos/ppo/r_mappo',
        './algos/ppo/r_mappo/algorithm', './algos/ppo/utils',
        '.nocturne/envs/', './nocturne_utils/', '.nocturne/python/', './build'
    ]
    src_dir = pathlib.Path(os.path.dirname(os.getcwd()))
    for dir in dirs_to_copy:
        copy_dir(dir, '*.py')
        copy_dir(dir, '*.yaml')

    return snap_dir


def main():
    """Launch experiments on SLURM cluster by overriding Hydra config."""
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--code_path',
                        default='/checkpoint/eugenevinitsky/nocturne')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    snap_dir = make_code_snap(args.experiment, args.code_path)
    print(str(snap_dir))
    overrides = Overrides()
    overrides.add('hydra/launcher', ['submitit_slurm'])
    overrides.add('hydra.launcher.partition', ['learnlab'])
    overrides.add('experiment', [args.experiment])
    # experiment parameters
    overrides.add('episode_length', [200])
    # algo
    overrides.add('algo', ['ppo'])
    overrides.add('algo.entropy_coef', [-0.001, 0.0, 0.001])
    overrides.add('algo.n_rollout_threads', [128])
    # rewards
    overrides.add('rew_cfg.goal_achieved_bonus', [10, 50])
    # misc
    overrides.add('scenario_path',
                  [PROJECT_PATH / 'scenarios/twenty_car_intersection.json'])

    cmd = [
        'python',
        str(snap_dir / 'code' / 'algos' / 'ppo' / 'nocturne_runner.py'), '-m'
    ]
    print(cmd)
    cmd += overrides.cmd()

    if args.dry:
        print(' '.join(cmd))
    else:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(snap_dir / 'code')
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()
