# `nocturne_lab`: fast driving simulator 🧪 + 🚗

`nocturne_lab` is a maintained fork of [Nocturne](https://github.com/facebookresearch/nocturne); a 2D, partially observed, driving simulator built in C++. Currently, `nocturne_lab` is used internally at the Emerge lab. You can get started with the intro examples 🏎️💨 [here](https://github.com/Emerge-Lab/nocturne_lab/tree/feature/nocturne_fork_cleanup/examples).

## Basic usage

```python
from nocturne.envs.base_env import BaseEnv

# Initialize an environment
env = BaseEnv(config=env_config)

# Reset
obs_dict = env.reset()

# Get info
agent_ids = [agent_id for agent_id in obs_dict.keys()]
dead_agent_ids = []

for step in range(1000):

    # Sample actions
    action_dict = {
        agent_id: env.action_space.sample() 
        for agent_id in agent_ids
        if agent_id not in dead_agent_ids
    }
    
    # Step in env
    obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

    # Update dead agents
    for agent_id, is_done in done_dict.items():
        if is_done and agent_id not in dead_agent_ids:
            dead_agent_ids.append(agent_id)

    # Reset if all agents are done
    if done_dict["__all__"]:
        obs_dict = env.reset()
        dead_agent_ids = []

# Close environment
env.close()
```

## Implemented algorithms

| Algorithm                              | Reference                                                  | Code  | Compatible with    | Test report                                                                                                                                                                  |
| -------------------------------------- | ---------------------------------------------------------- | ----- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PPO **single-agent** control | [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf) | [ppo_with_sb3.ipynb](https://github.com/Emerge-Lab/nocturne_lab/blob/feature/nocturne_fork_cleanup/examples/04_ppo_with_sb3.ipynb) | SB3 |        ✅ [Link](https://wandb.ai/daphnecor/single_agent_control_sb3_ppo/reports/Nocturne-with-SB3-s-PPO--Vmlldzo1NTg2Nzc4?accessToken=ednsze52absctmzw9sx28ry0y8uv4zt6nn4pre48tw7d2gwema0ayb5dj2zewwyp)                            |
| PPO **multi-agent** control  | [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf) | [05_ppo_with_sb3_ma_control.py](https://github.com/Emerge-Lab/nocturne_lab/blob/main/examples/05_ppo_with_sb3_ma_control.py) | SB3 | ✅ [Link](https://api.wandb.ai/links/daphnecor/uzgoj8de) | 
## Installation
The instructions for installing Nocturne locally are provided below. To use the package on a HPC (e.g. HPC Greene), follow the instructions in [./hpc/hpc_setup.md](./hpc/hpc_setup.md).

### Requirements

* Python (>=3.10)

### Virtual environment
Below different options for setting up a virtual environment are described. Either option works although `pyenv` is recommended.

> _Note:_ The virtual environment needs to be **activated each time** before you start working.

#### Option 1: `pyenv`
Create a virtual environment by running:

```shell
pyenv virtualenv 3.10.12 nocturne_lab
```

The virtual environment should be activated every time you start a new shell session before running subsequent commands:

```shell
pyenv shell nocturne_lab
```

Fortunately, `pyenv` provides a way to assign a virtual environment to a directory. To set it for this project, run:
```shell
pyenv local nocturne_lab
```

#### Option 2: `conda`
Create a conda environment by running:

```shell
conda env create -f ./environment.yml
```

This creates a conda environment using Python 3.10 called `nocturne_lab`.

To activate the virtual environment, run:

```shell
conda activate nocturne_lab
```

#### Option 3: `venv`
Create a virtual environment by running:

```shell
python -m venv .venv
```

The virtual environment should be activated every time you start a new shell session before running the subsequent command:

```shell
source .venv/bin/activate
```

### Dependencies

`poetry` is used to manage the project and its dependencies. Start by installing `poetry` in your virtual environment:

```shell
pip install poetry
```

Before installing the package, you first need to synchronise and update the git submodules by running:

```shell
# Synchronise and update git submodules
git submodule sync
git submodule update --init --recursive
```

Now install the package by running:

```shell
poetry install
```

> _Note_: If it fails to build `nocturne`, try running `poetry build` to get a more descriptive error message. One reason it fails may be because you don't have SFML installed, which can be done by running `brew install sfml` on mac or `sudo apt-get install libsfml-dev` on Linux.

---
> Under the hood the `nocturne` package uses the `nocturne_cpp` Python package that wraps the Nocturne C++ code base and provides bindings for Python to interact with the C++ code using `pybind11`.
---

### Common errors

- `KeyringLocked Failed to unlock the collection!`. Solution: first run `export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring` in your terminal, then rerun `poetry install` [stackOverflow with more info](https://stackoverflow.com/questions/74438817/poetry-failed-to-unlock-the-collection)

### Development setup
To configure the development setup, run:
```shell
# Install poetry dev dependencies
poetry install --only=dev

# Install pre-commit (for flake8, isort, black, etc.)
pre-commit install

# Optional: Install poetry docs dependencies
poetry install --only=docs
```

## Ongoing work

Here is a list of features that we are developing:

- @Daphne: Support for SB3's PPO algorithm with multi-agent control
- @Alex: Logging and unit testing
- @Tiyas: Sample efficient RL by prioritized scene replay
