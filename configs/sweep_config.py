sweep_config_arch = {
    'project': 'test',
    'name': 'mlp_architectures',
    'metric': {'name': 'rollout/ep_rew_mean_norm', 'goal':'maximize'},
    'method': 'grid',
    'parameters': {
        'total_timesteps': {'values': [10_000_000]},
        'policy_layers': {'values': [[1048, 512, 128], [2096, 1048, 512, 64]]},
        'normalize_state': {'values': [True, False]},
        'policy_act_func': {'values': ['tanh', 'relu']},
        'memory': {'values': [1]},
        'ent_coef': {'values': [0, 0.025]},
        'vf_coef': {'values': [0.5, 0.25]},
        'num_controlled_agents': {'values': [2]},
        'seed': {'values': [42]},
    }
}

sweep_config2 = {
    'project': 'nocturne_ppo_exp',
    'name':'ppo_sweep_single_scene',
    'metric': {'name': 'rollout/ep_rew_mean_norm', 'goal':'maximize'},
    'method': 'random',
    'parameters': {
        'total_timesteps': {'values': [100_000_000]},
        'seed': {'values': [0, 2]},
        'ent_coef': {'values': [0, 0.025, 0.05]},
        'vf_coef': {'values': [0.25, 0.5]},
        'num_controlled_agents': {'values': [3]},
    }
}
