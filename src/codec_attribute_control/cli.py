"""Hydra entry point; all experiment parameters are YAML overrides."""
import json

import hydra
from omegaconf import DictConfig, OmegaConf

from .experiment import intervene, train


@hydra.main(version_base=None, config_path="configs", config_name="experiment")
def main(cfg: DictConfig) -> None:
    if cfg.task == "train":
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
    elif cfg.task == "intervene":
        if cfg.intervention.audio is None:
            raise ValueError("Set intervention.audio=/path/to/recording.wav")
        result = intervene(
            cfg.intervention.audio,
            cfg.probe.output,
            cfg.intervention.output_dir,
            shifts=list(cfg.intervention.semitones),
            device=cfg.device,
        )
    else:
        raise ValueError(f"Unsupported task: {cfg.task}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
