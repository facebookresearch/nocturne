# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run script that generates summary statistics for a folder of IL agents."""
import json
import os

import numpy as np
import torch

from nocturne.utils.eval.average_displacement import compute_average_displacement
from cfgs.config import PROCESSED_VALID_NO_TL, PROJECT_PATH

if __name__ == '__main__':
    outer_model_folder = '/checkpoint/eugenevinitsky/nocturne/sweep/imitation/2022.06.13/arxiv_il_v4_1kf/18.49.39'
    models = []
    cfg_dicts = []
    for (dirpath, dirnames, filenames) in os.walk(outer_model_folder):
        if 'configs.json' in filenames:
            with open(os.path.join(dirpath, 'configs.json'), 'r') as file:
                cfg_dict = json.load(file)
            # now snag the model with the largest checkpoint
            max_val = -100
            cur_model_name = None
            for file in filenames:
                if '.pth' in file:
                    checkpoint_val = int(file.split('.')[0].split('_')[-1])
                    if checkpoint_val > max_val:
                        max_val = checkpoint_val
                        cur_model_name = file
            cfg_dicts.append(cfg_dict)
            model = torch.load(os.path.join(dirpath, cur_model_name)).to('cpu')
            model.actions_grids = [x.to('cpu') for x in model.actions_grids]
            model.eval()
            model.nn[0].eval()
            models.append(model)
    results = np.zeros((len(cfg_dicts), 8))
    for i, (cfg_dict, model) in enumerate(zip(cfg_dicts, models)):
        ade, fde, collisions, goals = compute_average_displacement(
            PROCESSED_VALID_NO_TL, model=model, configs=cfg_dict)
        results[i, 0] = ade[0]
        results[i, 1] = ade[1]
        results[i, 2] = fde[0]
        results[i, 3] = fde[1]
        results[i, 4] = collisions[0]
        results[i, 5] = collisions[1]
        results[i, 6] = goals[0]
        results[i, 7] = goals[1]
    np.save(os.path.join(PROJECT_PATH, 'scripts/paper_plots/il_results.npy'),
            results)
    print(
        f'ade {np.mean(results[:, 0])} ± {np.std(results[:, 0]) / np.sqrt(results[:, 0].shape[0])}'
    )
    print(
        f'fde {np.mean(results[:, 2])} ± {np.std(results[:, 2]) / np.sqrt(results[:, 0].shape[0])}'
    )
    print(
        f'collisions {np.mean(results[:, 4])} ± {np.std(results[:, 4]) / np.sqrt(results[:, 0].shape[0])}'
    )
    print(
        f'goals {np.mean(results[:, 6])} ± {np.std(results[:, 6]) / np.sqrt(results[:, 0].shape[0])}'
    )
