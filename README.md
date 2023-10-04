# `nocturne_lab`: fast driving simulator 🧪 + 🚗

`nocturne_lab` is a maintained fork of [Nocturne](https://github.com/facebookresearch/nocturne), which is a 2D, partially observed, driving simulator built in C++. `nocturne_lab` is currently only used internally at the Emerge lab.

## Basic usage

```python
from nocturne.envs.base_env import BaseEnv

# Initialize an environment
env = BaseEnv(config=env_config)

# Reset
obs_dict = env.reset()
num_agents = len(env.controlled_agents)

# Step through env
for _ in range(1000):

    # Take action(s)
    action_dict = {
        agent_id: env.action_space.sample()
        for agent_id in agent_ids
        if agent_id not in dead_agent_ids
    }

    # Step
    obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

    if done_dict["__all__"]:
        obs_dict = env.reset()

env.close()
```

---
> 🚀 **New here?** Get started with the intro [examples](https://github.com/Emerge-Lab/nocturne_lab/tree/feature/nocturne_fork_cleanup/examples)
---

## Implemented algorithms

| Algorithm                              | Reference                                                  | Code  | Compatible with    | Notes                                                                                                                                                                  |
| -------------------------------------- | ---------------------------------------------------------- | ----- | ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| PPO **single-agent** control | [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf) | #TODO | Stable baselines 3 |                                                                                                                                                                        |
| PPO **multi-agent** control  | [Schulman et al., 2017](https://arxiv.org/pdf/1707.06347.pdf) | #TODO | Stable baselines 3 | SB3 doesn't support multi-agent environments. Using the `VecEnv`class to treat observations from multiple agents as a set of vectorized single-agent environments. |
|                                        |                                                            |       |                    |                                                                                                                                                                        |
|                                        |                                                            |       |                    |                                                                                                                                                                        |

## Installation

#TODO

## Ongoing work

Here is a list of features that we are developing:

#TODO
