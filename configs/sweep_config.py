sweep_config_arch = {
    'project': 'test',
    'name': 'mlp_architectures_normalized',
    'metric': {'name': 'rollout/ep_rew_mean_norm', 'goal':'maximize'},
    'method': 'random',
    'parameters': {
        'total_timesteps': {'values': [10_000_000]},
        'policy_layers': {'values': [[64, 64], [512, 256, 64], [1048, 512, 128]]},
        'normalize_state': {'values': ['true', 'false']},
        'policy_act_func': {'values': ['tanh', 'relu']},
        'memory': {'values': [1]},
        'ent_coef': {'values': [0, 0.025, 0.05]},
        'vf_coef': {'values': [0.5, 0.25]},
        'num_controlled_agents': {'values': [5]},
        'seed': {'values': [42]},
    }
}

