"""Download trained policies from W&B."""

import wandb


if __name__ == "__main__":

    ROOT = "models/hr_rl"

    entity = "daphnecor"
    project = "scaling_ppo"
    collection_name = "nocturne-hr-ppo-01_02_11_16"
    
    # Always initialize a W&B run to start tracking
    wandb.init()

    # Download model version files
    path = wandb.use_artifact(f"{entity}/{project}/{collection_name}:latest").download(
        root=ROOT,
    )

    print('Downloaded model files to: ', path)