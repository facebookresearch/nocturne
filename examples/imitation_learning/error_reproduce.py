# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Imitation learning training script (behavioral cloning)."""
from datetime import datetime
from pathlib import Path
import pickle
import random
import json

import hydra
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from examples.imitation_learning.model import ImitationAgent
from examples.imitation_learning.waymo_data_loader import WaymoDataset


def set_seed_everywhere(seed):
    """Ensure determinism."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="../../cfgs/imitation", config_name="config")
def main(args):
    """Train an IL model."""
    set_seed_everywhere(args.seed)
    expert_bounds = [[-6, 6], [-0.7, 0.7]]
        
    # load dataloader config
    dataloader_config = {
        'tmin': 0,
        'tmax': 90,
        'view_dist': 80,
        'view_angle': np.radians(120),
        'dt': 0.1,
        'expert_action_bounds': None,
        'expert_position': True,
        'state_normalization': 100,
        'n_stacked_states': 5,
        'perturbations': False,
    }

    scenario_config = {
        'start_time': 0,
        'allow_non_vehicles': True,
        'spawn_invalid_objects': True,
        'max_visible_road_points': 500,
        'sample_every_n': 1,
        'road_edge_first': False,
    }
    
    dataset = WaymoDataset(
        data_path=args.path,
        file_limit=args.num_files,
        dataloader_config=dataloader_config,
        scenario_config=scenario_config,
    )
    data_loader = iter(
        DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.n_cpus,
            pin_memory=True,
        ))

    # create exp dir
    time_str = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    exp_dir = Path.cwd() / Path('train_logs') / time_str
    exp_dir.mkdir(parents=True, exist_ok=True)

    # train loop
    for epoch in range(args.epochs):
        print(f'\nepoch {epoch+1}/{args.epochs}')
        n_samples = epoch * args.batch_size * (args.samples_per_epoch //
                                               args.batch_size)

        for i in tqdm(range(args.samples_per_epoch // args.batch_size),
                      unit='batch'):
            # get states and expert actions
            states, expert_actions = next(data_loader)


if __name__ == '__main__':
    main()