"""Dedicated Hydra entrypoint reusing the shared QC training pipeline."""
import hydra
from .train import run_training


@hydra.main(version_base=None, config_path="configs", config_name="train_codes")
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
