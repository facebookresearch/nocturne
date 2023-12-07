"""Configuration manager."""

import yaml
from box import Box


def load_config(cfg: str) -> Box:
    """Load configurations as a Box object.
    Args:
        cfg (str): Name of config file.

    Returns:
        Box: Box representation of configurations.
    """
    with open(f"./configs/{cfg}.yaml", "r") as stream:
        config = Box(yaml.safe_load(stream))
    return config


def load_config_nb(cfg: str) -> Box:
    """Load configurations as a Box object.
    Args:
        cfg (str): Name of config file.

    Returns:
        Box: Box representation of configurations.
    """
    with open(f"../configs/{cfg}.yaml", "r") as stream:
        config = Box(yaml.safe_load(stream))
    return config