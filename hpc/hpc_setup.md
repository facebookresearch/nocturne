## Setup of `nocturne_lab` on HPC

### Installation (only do the first time)

1. Clone repo into HPC Greene at location `/scratch/$USER`, where `$USER` is your netid.
2. Move into `nocturne_lab` folder by running: `cd nocturne_lab`
3. Set up Singularity image: `bash ./hpc/launch_image.sh`
4. Create a virtual Python environment: `python3 -m venv .venv`
5. Activate venv: `source .venv/bin/activate`
6. Install Nocturne: `bash ./hpc/post_setup_image.sh`
7. Restart Singularity image: `exit` 
8. Relaunch image: `bash ./hpc/launch_image.sh`
9. Check if Nocturne is properly installed: 
    - (a) launch a Python shell
    - (b) import Nocturne by running `import nocturne; import nocturne_cpp`

### Usage

Request an interactive compute node:
```shell
# Example: Request a single GPU for one hour
srun --nodes=1 --tasks-per-node=1 --cpus-per-task=1 --mem=10GB --gres=gpu:1 --time=1:00:00 --pty /bin/bash
```

Navigate to the repo:
```shell
cd /scratch/$USER/nocturne_lab
```

Launch the Singularity image:
```shell
bash ./hpc/launch_image.sh
```

Activate the Python virtual environment:
```shell
source .venv/bin/activate
```