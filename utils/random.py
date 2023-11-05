"""Collection of utilty functions for random number generation and seeding."""

import random

import numpy as np
import torch
from box import Box


def init_seed(exp_config: Box) -> None:
    """Set the seed for a RL experiment.

    Args:
        exp_config (Box): Experiment configurations.
    """
    random.seed(exp_config.seed)
    np.random.seed(exp_config.seed)
    torch.manual_seed(exp_config.seed)
    torch.backends.cudnn.deterministic = exp_config.torch_deterministic


def set_seed_everywhere(seed):
    """Ensure determinism IL setting."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)