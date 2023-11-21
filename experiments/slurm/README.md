## How to run many jobs on the cluster

### Step 1. Generate an sbatch and execution script 

Navigate to `sbatch_generator.py` and define field and sweep params:
```Python
# Define SBATCH params
fields = {
    'time_h': 10, # Time per job
    'num_gpus': 1, # GPUs per job 
    'max_sim_jobs': 25, 
}

# Define sweep conf
params = {
    'sweep_name': ['sweep_act_space_indiv'], # Project name
    'steer_disc': [5, 9, 15], # Action space; 5 is the default
    'accel_disc': [5, 9, 15], # Action space; 5 is the default
    'ent_coef' : [0, 0.025, 0.05],   # Entropy coefficient in the policy loss
    'vf_coef'  : [0.5, 0.25], # Value coefficient in the policy loss
    'seed' : [8, 42, 3], # Random seed
    'activation_fn': ['tanh', 'relu'],
    'num_files': [1],
}

save_scripts(
    sbatch_filename="sbatch_sweep.sh",
    bash_filename="bash_exec.sh", 
    file_path="experiments/slurm/run_scripts/",
    run_script="experiments/rl/ppo_w_cli_args.py",
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

Run
```Python
sbatch experiments/slurm/run_scripts/sbatch_sweep.sh
```
to submit the job arrays. This script will call the `bash_exec.sh` for each array index.


## Testing

Load python for testing (no need to request an interactive node)
```
module load python/intel/3.8.6
```
NOTE: Make sure to run the jobs in a new terminal, otherwise it will use this Python version.


## `TODOs`

- Set paths to python script and bash exec script using variables (in `sbatch_generator.py`)
- Scale up setup such that every defined sweep gets a unique id


## Run ppo via the command line with args


Check the arguments of the script using `--help`:

```shell
(.venv) Singularity> python experiments/rl/ppo_w_cli_args.py --help
                                                                                                                                                                                                                                               
 Usage: ppo_w_cli_args.py [OPTIONS]                                                                                                                                                                                                                          
                                                                                                                                                                                                                                                             
 Train RL agent using PPO with CLI arguments.                                                                                                                                                                                                                
                                                                                                                                                                                                                                                             
╭─ Options ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --steer-disc           INTEGER  [default: 5]                                                                                                                                                                                                              │
│ --accel-disc           INTEGER  [default: 5]                                                                                                                                                                                                              │
│ --ent-coef             FLOAT    [default: 0.0]                                                                                                                                                                                                            │
│ --vf-coef              FLOAT    [default: 0.5]                                                                                                                                                                                                            │
│ --seed                 INTEGER  [default: 42]                                                                                                                                                                                                             │
│ --policy-layers        INTEGER  [default: 64, 64]                                                                                                                                                                                                         │
│ --activation-fn        TEXT     [default: tanh]                                                                                                                                                                                                           │
│ --help                          Show this message and exit.                                                                                                                                                                                               │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

