# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run rllib experiments on a SLURM cluster."""
import argparse
import os
import pathlib
import shutil
from datetime import datetime
from subprocess import Popen

from cfgs.config import PROJECT_PATH
from scripts.utils import Overrides


def make_code_snap(experiment, code_path, str_time):
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
    if len(code_path) > 0:
        snap_dir = pathlib.Path(code_path)
    else:
        snap_dir = pathlib.Path.cwd()
    snap_dir /= str_time
    snap_dir /= f'{experiment}'
    snap_dir.mkdir(exist_ok=True, parents=True)

    def copy_dir(dir, pat):
        dst_dir = snap_dir / 'code' / dir
        dst_dir.mkdir(exist_ok=True, parents=True)
        for f in (src_dir / dir).glob(pat):
            shutil.copy(f, dst_dir / f.name)

    dirs_to_copy = [
        '.', './cfgs/', './examples/', './cfgs/algorithm', './envs/',
        './nocturne_utils/', './python/', './scenarios/', './build'
    ]
    src_dir = pathlib.Path(PROJECT_PATH)
    for dir in dirs_to_copy:
        copy_dir(dir, '*.py')
        copy_dir(dir, '*.yaml')

    return snap_dir


def main():
    """Launch experiments on SLURM cluster by overriding Hydra config."""
    username = os.environ["USER"]
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument(
        '--code_path',
        default=f'/checkpoint/{username}/nocturne/sample_factory_runs')
    parser.add_argument('--dry', action='store_true')
    args = parser.parse_args()

    now = datetime.now()
    str_time = now.strftime('%Y.%m.%d_%H%M%S')
    snap_dir = make_code_snap(args.experiment, args.code_path, str_time)
    overrides = Overrides()
    overrides.add('hydra/launcher', ['ray'])
    overrides.add('hydra.launcher.partition', ['learnlab'])

    cmd = [
        'python',
        str(snap_dir / 'code' / 'examples' / 'run_rllib.py'), '-m'
    ]
    cmd += overrides.cmd()
    print(cmd)

    if args.dry:
        print(' '.join(cmd))
    else:
        env = os.environ.copy()
        env['PYTHONPATH'] = str(snap_dir / 'code')
        p = Popen(cmd, env=env)
        p.communicate()


if __name__ == '__main__':
    main()
