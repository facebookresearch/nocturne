sweep_config_arch = {
    'project': 'ppo_scaling_2_scenes',
    'name': 'mlp_archs',
    'metric': {'name': 'rollout/ep_rew_mean_norm', 'goal':'maximize'},
    'method': 'random',
    'parameters': {
        'total_timesteps': {'values': [10_000_000]},
        'policy_layers': {'values': [[512, 256, 64], [64, 64]]},
        'policy_act_func': {'values': ['tanh', 'relu']},
        'memory': {'values': [1]},
        'ent_coef': {'values': [0, 0.025, 0.05]},
        'vf_coef': {'values': [0.5]},
        'num_controlled_agents': {'values': [5]},
        'seed': {'values': [42, 6]},
    }
}

