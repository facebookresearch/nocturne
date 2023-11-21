
sweep_name_values=( sweep_hr_ppo )
steer_disc_values=( 5 7 )
accel_disc_values=( 5 )
ent_coef_values=( 0 0.001 )
vf_coef_values=( 0.5 )
seed_values=( 8 42 )
policy_size_values=( small )
policy_arch_values=( mlp )
num_files_values=( 10 )
total_timesteps_values=( 45000000 )
reg_weight_values=( 0.0 0.05 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 )

trial=${SLURM_ARRAY_TASK_ID}
sweep_name=${sweep_name_values[$(( trial % ${#sweep_name_values[@]} ))]}
trial=$(( trial / ${#sweep_name_values[@]} ))
steer_disc=${steer_disc_values[$(( trial % ${#steer_disc_values[@]} ))]}
trial=$(( trial / ${#steer_disc_values[@]} ))
accel_disc=${accel_disc_values[$(( trial % ${#accel_disc_values[@]} ))]}
trial=$(( trial / ${#accel_disc_values[@]} ))
ent_coef=${ent_coef_values[$(( trial % ${#ent_coef_values[@]} ))]}
trial=$(( trial / ${#ent_coef_values[@]} ))
vf_coef=${vf_coef_values[$(( trial % ${#vf_coef_values[@]} ))]}
trial=$(( trial / ${#vf_coef_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
trial=$(( trial / ${#seed_values[@]} ))
policy_size=${policy_size_values[$(( trial % ${#policy_size_values[@]} ))]}
trial=$(( trial / ${#policy_size_values[@]} ))
policy_arch=${policy_arch_values[$(( trial % ${#policy_arch_values[@]} ))]}
trial=$(( trial / ${#policy_arch_values[@]} ))
num_files=${num_files_values[$(( trial % ${#num_files_values[@]} ))]}
trial=$(( trial / ${#num_files_values[@]} ))
total_timesteps=${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}
trial=$(( trial / ${#total_timesteps_values[@]} ))
reg_weight=${reg_weight_values[$(( trial % ${#reg_weight_values[@]} ))]}

source /scratch/dc4971/nocturne_lab/.venv/bin/activate
python experiments/hr_rl/run_hr_ppo_w_cli_args.py --sweep-name=${sweep_name} --steer-disc=${steer_disc} --accel-disc=${accel_disc} --ent-coef=${ent_coef} --vf-coef=${vf_coef} --seed=${seed} --policy-size=${policy_size} --policy-arch=${policy_arch} --num-files=${num_files} --total-timesteps=${total_timesteps} --reg-weight=${reg_weight}
