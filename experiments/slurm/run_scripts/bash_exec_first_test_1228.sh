
sweep_name_values=( first_test_1228 )
ent_coef_values=( 0 0.001 )
vf_coef_values=( 0.5 )
seed_values=( 42 )
dropout_values=( 0.0 0.05 )
arch_road_objects_values=( tiny small )
arch_road_graph_values=( tiny small )
arch_shared_net_values=( tiny small medium )
activation_fn_values=( tanh relu )
num_files_values=( 10 )
total_timesteps_values=( 30000000 )

trial=${SLURM_ARRAY_TASK_ID}
sweep_name=${sweep_name_values[$(( trial % ${#sweep_name_values[@]} ))]}
trial=$(( trial / ${#sweep_name_values[@]} ))
ent_coef=${ent_coef_values[$(( trial % ${#ent_coef_values[@]} ))]}
trial=$(( trial / ${#ent_coef_values[@]} ))
vf_coef=${vf_coef_values[$(( trial % ${#vf_coef_values[@]} ))]}
trial=$(( trial / ${#vf_coef_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
trial=$(( trial / ${#seed_values[@]} ))
dropout=${dropout_values[$(( trial % ${#dropout_values[@]} ))]}
trial=$(( trial / ${#dropout_values[@]} ))
arch_road_objects=${arch_road_objects_values[$(( trial % ${#arch_road_objects_values[@]} ))]}
trial=$(( trial / ${#arch_road_objects_values[@]} ))
arch_road_graph=${arch_road_graph_values[$(( trial % ${#arch_road_graph_values[@]} ))]}
trial=$(( trial / ${#arch_road_graph_values[@]} ))
arch_shared_net=${arch_shared_net_values[$(( trial % ${#arch_shared_net_values[@]} ))]}
trial=$(( trial / ${#arch_shared_net_values[@]} ))
activation_fn=${activation_fn_values[$(( trial % ${#activation_fn_values[@]} ))]}
trial=$(( trial / ${#activation_fn_values[@]} ))
num_files=${num_files_values[$(( trial % ${#num_files_values[@]} ))]}
trial=$(( trial / ${#num_files_values[@]} ))
total_timesteps=${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}

source /scratch/dc4971/nocturne_lab/.venv/bin/activate
python experiments/hr_rl/run_hr_ppo_cli.py --sweep-name=${sweep_name} --ent-coef=${ent_coef} --vf-coef=${vf_coef} --seed=${seed} --dropout=${dropout} --arch-road-objects=${arch_road_objects} --arch-road-graph=${arch_road_graph} --arch-shared-net=${arch_shared_net} --activation-fn=${activation_fn} --num-files=${num_files} --total-timesteps=${total_timesteps}
