"""Hydra entry point; QC owns dataset loading and training."""
from pathlib import Path

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from quick_convert.pipelines.training.pipeline import TrainingPipeline
from quick_convert.utils.resolvers import register_config_resolvers

register_config_resolvers()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    train_dataset = instantiate(cfg.dataset, splits=[cfg.train_split], load=cfg.f0.dataset_load, resource_providers=instantiate(cfg.f0.resource_providers))
    val_dataset = instantiate(cfg.dataset, splits=[cfg.val_split], load=cfg.f0.dataset_load, resource_providers=instantiate(cfg.f0.resource_providers)) if cfg.get("val_split") is not None else None
    trainer = instantiate(cfg.trainer)
    pipeline = TrainingPipeline(
        trainer=trainer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        out_dir=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
    )
    pipeline.prepare()
    pipeline.write_config(OmegaConf.to_yaml(cfg, resolve=True))
    pipeline.log_config(OmegaConf.to_container(cfg, resolve=True))
    pipeline.run()


if __name__ == "__main__":
    main()
