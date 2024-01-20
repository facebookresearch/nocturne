"""Download trained policies from W&B."""

import wandb

if __name__ == "__main__":
    ROOT = "models/hr_rl"

    entity = "daphnecor"
    project = "hr_rl_exp"
    run_id = "01_08_06_29"

    collection_name = f"nocturne-hr-ppo-{run_id}"

    # Always initialize a W&B run to start tracking
    wandb.init()

    # Download model version files
    path = wandb.use_artifact(f"{entity}/{project}/{collection_name}:latest").download(
        root=ROOT,
    )

    print("Downloaded model files to: ", path)
