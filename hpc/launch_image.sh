# Script for launching the singularity image for Nocturne (lab version)
# ---
# Note: Script must be run from the Nocturne LAB repository (i.e. /scratch/$USER/nocturne_lab/)
# and executed each time you want to use Nocturne (lab version) on the HPC (e.g. Greene).

# Constants
PROJECT="Nocturne (lab version)"
PROJECT_DOCKER=docker://daphnecor/nocturne
SINGULARITY_IMAGE=./hpc/nocturne.sif
OVERLAY_LOC=/scratch/work/public/overlay-fs-ext3
OVERLAY_FILE=overlay-15GB-500K.ext3

# Check if singularity image exists, if not pull Singularity image from Docker Hub
if [ ! -f "${SINGULARITY_IMAGE}" ]; then
    echo "Pulling Docker container from ${PROJECT_DOCKER}"
    singularity pull $SINGULARITY_IMAGE $PROJECT_DOCKER
fi

# Check if overlay file exists, if not create it
if [ ! -f "${OVERLAY_FILE}" ]; then  # Overlay file does not exist
    echo "Setting up ${PROJECT_DOCKER} with initial overlay ${OVERLAY_FILE}.gz"

    if [ ! -f "${OVERLAY_FILE}.gz" ]; then  # Overlay file has not been copied yet
        echo "Copying overlay ${OVERLAY_FILE}.gz from ${OVERLAY_LOC}..."
        cp -rp "${OVERLAY_LOC}/${OVERLAY_FILE}.gz" . -n
        echo "Unzipping overlay ${OVERLAY_FILE}.gz..."
        gunzip "${OVERLAY_FILE}.gz" -n
    fi

    # Launch singularity for the first time
    echo 'Launching singularity image in WRITE (edit) mode...'

    # Welcome message
    echo "Run the following to initialize ${PROJECT}:"
    echo "  (1) create a virtual Python environment: 'python3 -m venv .venv'"
    echo "  (2) activate venv: 'source .venv/bin/activate'"
    echo "  (3) install Nocturne: 'bash ./hpc/post_setup_image.sh'"

    # Launch singularity image in write mode
    singularity exec --nv --overlay "${OVERLAY_FILE}:rw" \
        "${SINGULARITY_IMAGE}" \
        /bin/bash

else  # Overlay Singularity image and overlay file exist

    # Launch singularity
    echo 'Launching singularity image in OVERLAY (use) mode...'

    # Welcome message
    echo "Run the following to activate the Python environment:"
    echo "  (1) activate venv: 'source .venv/bin/activate'"

    # Launch singularity image in use mode
    singularity exec --nv --overlay "${OVERLAY_FILE}:rw" \
        "${SINGULARITY_IMAGE}" \
        /bin/bash

fi
