
sweep_name_values=( sweep_act_space_indiv )
steer_disc_values=( 5 )
accel_disc_values=( 5 )
ent_coef_values=( 0 )
vf_coef_values=( 0.5 )
seed_values=( 8 )
activation_fn_values=( tanh )
num_files_values=( 1 )
total_timesteps_values=( 10000 )

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
activation_fn=${activation_fn_values[$(( trial % ${#activation_fn_values[@]} ))]}
trial=$(( trial / ${#activation_fn_values[@]} ))
num_files=${num_files_values[$(( trial % ${#num_files_values[@]} ))]}
trial=$(( trial / ${#num_files_values[@]} ))
total_timesteps=${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}

source /scratch/dc4971/nocturne_lab/.venv/bin/activate && python experiments/rl/ppo_w_cli_args.py --sweep-name=${sweep_name} --steer-disc=${steer_disc} --accel-disc=${accel_disc} --ent-coef=${ent_coef} --vf-coef=${vf_coef} --seed=${seed} --activation-fn=${activation_fn} --num-files=${num_files} --total-timesteps=${total_timesteps}
