## How to run many jobs on the cluster

### Step 1. Generate an sbatch and execution script 

Navigate to `sbatch_generator.py` and define field and sweep params:
```Python
# Define SBATCH params
fields = {
    'time_h': 5, # Max time per job
    'num_gpus': 1, # Number of gpus per job
    'max_sim_jobs': 25, # Max jobs to be run simultaneously
}

# Define sweep conf
params = {
    'steer_disc': [5, 9, 15], # Action space; 5 is the default
    'accel_disc': [5, 9, 15], # Action space; 5 is the default
    'ent_coef' : [0, 0.025, 0.05], # Entropy coefficient in the policy loss
    'vf_coef'  : [0.5, 0.25], # Value coefficient in the policy loss
    'seed' : [42, 8], # Random seed
    'policy_layers': [[64, 64], [512, 256, 64], [1024, 512, 256, 64]] # Size of the policy network
}

# Save 
save_scripts(
    sbatch_filename="sbatch_test.sh",
    bash_filename="bash_exec.sh", #NOTE: don't change this name
    file_path="experiments/slurm/run_scripts/",
    fields=fields,
    params=params,
)
```

Then run
```
python experiments/slurm/sbatch_generator.py
```

this will save an sbatch and and execute script.

### Step 2: Submit the jobs

The jobs using
```Python
sbatch path_to_your_sbatch_script.sh
```


### Notes

Load python for testing
```
module load python/intel/3.8.6
```


### DO

