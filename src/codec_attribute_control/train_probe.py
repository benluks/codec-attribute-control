"""Dedicated Hydra entry point for probe training.

Training implementation is still the legacy prototype; the upcoming
BaseTrainingModule/LightningTrainer/TrainingPipeline refactor will replace it.
"""
import json

import hydra
from omegaconf import DictConfig

from .experiment import train


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg: DictConfig) -> None:
    result = train(
        cfg.dataset.root,
        cfg.probe.output,
        backend=cfg.f0.backend,
        alpha=cfg.probe.alpha,
        seed=cfg.seed,
        max_recordings=cfg.dataset.max_recordings,
        checkpoint_path=cfg.f0.checkpoint_path,
        device=cfg.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
