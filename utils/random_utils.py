"""Collection of utilty functions for random number generation and seeding."""

import random
import numpy as np
import torch
from box import Box


def init_seed(env_config:Box, exp_config: Box, seed) -> None:
    """Set the seed for a RL experiment.

    Args:
        env_config (Box): Environment configurations.
        exp_config (Box): Experiment configurations.
    """
    env_config.seed = seed
    exp_config.seed = seed
    random.seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    torch.manual_seed(exp_config.seed)


def set_seed_everywhere(seed):
    """Ensure determinism IL setting."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)