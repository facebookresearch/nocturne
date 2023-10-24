sweep_config = {
    'project': 'nocturne_ppo_exp',
    'name':'ppo_sweep_single_scene',
    'metric': {'name': 'rollout/ep_rew_mean_norm', 'goal':'maximize'},
    'method': 'grid',
    'parameters': {
        'total_timesteps': {'values': [10_000]},
        'seed': {'values': [6, 43, 42]},
        'ent_coef': {'values': [0, 0.025, 0.05]},
        'vf_coef': {'values': [0.25, 0.5]},
        'num_controlled_agents': {'values': [1]},
    }
}