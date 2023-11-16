#!/bin/bash

#SBATCH --array=0-0%25
#SBATCH --job-name=my_job
#SBATCH --output=experiments/slurm/logs/output_%A_%a.txt
#SBATCH --error=experiments/slurm/logs/error_%A_%a.txt
#SBATCH --mem=10GB
#SBATCH --time=0-1:0:0
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

SINGULARITY_IMAGE=hpc/nocturne.sif
OVERLAY_FILE=hpc/overlay-15GB-500K.ext3

singularity exec --nv --overlay "${OVERLAY_FILE}:ro"     "${SINGULARITY_IMAGE}"     /bin/bash experiments/slurm/run_scripts/bash_exec.sh "${SLURM_ARRAY_TASK_ID}"
echo "Successfully launched image."