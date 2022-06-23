# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Set path to all the Waymo data and the parsed Waymo files."""
import os
from pathlib import Path

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

VERSION_NUMBER = 2

PROJECT_PATH = Path.resolve(Path(__file__).parent.parent)
DATA_FOLDER = '/checkpoint/eugenevinitsky/waymo_open/motion_v1p1/uncompressed/scenario/'
TRAIN_DATA_PATH = os.path.join(DATA_FOLDER, 'training')
VALID_DATA_PATH = os.path.join(DATA_FOLDER, 'validation')
TEST_DATA_PATH = os.path.join(DATA_FOLDER, 'testing')
PROCESSED_TRAIN_NO_TL = os.path.join(
    DATA_FOLDER, f'formatted_json_v{VERSION_NUMBER}_no_tl_train')
PROCESSED_VALID_NO_TL = os.path.join(
    DATA_FOLDER, f'formatted_json_v{VERSION_NUMBER}_no_tl_valid')
PROCESSED_TRAIN = os.path.join(DATA_FOLDER,
                               f'formatted_json_v{VERSION_NUMBER}_train')
PROCESSED_VALID = os.path.join(DATA_FOLDER,
                               f'formatted_json_v{VERSION_NUMBER}_valid')
ERR_VAL = -1e4


def get_scenario_dict(hydra_cfg):
    """Convert the `scenario` key in the hydra config to a true dict."""
    if isinstance(hydra_cfg['scenario'], dict):
        return hydra_cfg['scenario']
    else:
        return OmegaConf.to_container(hydra_cfg['scenario'], resolve=True)


def get_default_scenario_dict():
    GlobalHydra.instance().clear()
    initialize(config_path="./")
    cfg = compose(config_name="config")
    return get_scenario_dict(cfg)
