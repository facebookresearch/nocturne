#!/bin/bash

########## SBATCH COMMANDS ##########

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sweep
#SBATCH --output=out_%A_%a.txt
#SBATCH --error=error_%A_%a.err
#SBATCH --array=1-5%5
#SBATCH --time=0-1:00:00
#SBATCH --mem=10GB

########## SBATCH COMMANDS ##########

echo "Currently running array index: " $SLURM_ARRAY_TASK_ID

# Define variables and index using $SLURM_ARRAY_TASK_ID

# TEMPLATE SCRIPT: Launch singularity image
SINGULARITY_IMAGE=hpc/nocturne.sif
OVERLAY_FILE=hpc/overlay-15GB-500K.ext3

singularity exec --nv --overlay "${OVERLAY_FILE}:ro" \
    "${SINGULARITY_IMAGE}" \ 
    /bin/bash slurm_exec_script.sh $SLURM_ARRAY_TASK_ID
echo "Successfully launched image."