
source .venv/bin/activate

steer_disc_values=( 5 )
accel_disc_values=( 5 9 )

trial=${SLURM_ARRAY_TASK_ID}
steer_disc=${steer_disc_values[$(( trial % ${#steer_disc_values[@]} ))]}
trial=$(( trial / ${#steer_disc_values[@]} ))
accel_disc=${accel_disc_values[$(( trial % ${#accel_disc_values[@]} ))]}

python experiments/rl/ppo_w_cli_args.py --steer-disc=${steer_disc} --accel-disc=${accel_disc}
