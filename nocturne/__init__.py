# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Import file for Nocturne objects."""
from nocturne_cpp import (Action, CollisionType, ObjectType, Object, RoadLine,
                          RoadType, Scenario, Simulation, Vector2D, Vehicle)

__all__ = [
    "Action",
    "CollisionType",
    "ObjectType",
    "Object",
    "RoadLine",
    "RoadType",
    "Scenario",
    "Simulation",
    "Vector2D",
    "Vehicle",
    "envs",
]

import os
from cfgs.config import PROCESSED_TRAIN_NO_TL, PROCESSED_VALID_NO_TL

os.environ["PROCESSED_TRAIN_NO_TL"] = str(PROCESSED_TRAIN_NO_TL)
os.environ["PROCESSED_VALID_NO_TL"] = str(PROCESSED_VALID_NO_TL)
