"""Hydra entry point; QC owns dataset loading and training."""
from importlib.resources import files
from pathlib import Path

import hydra
from hydra.core.config_store import ConfigStore
from hydra.utils import instantiate
from omegaconf import OmegaConf
from quick_convert.pipelines.training.pipeline import TrainingPipeline
from quick_convert.utils.resolvers import register_config_resolvers

register_config_resolvers()
ConfigStore.instance().store(
    group="dataset",
    name="librispeech",
    node=OmegaConf.load(files("quick_convert").joinpath("configs/dataset/librispeech.yaml")),
)


def run_training(cfg):
    providers = instantiate(cfg.f0.resource_providers)
    if "codes" in cfg:
        providers = [*providers, *instantiate(cfg.codes.resource_providers)]

    def dataset(split):
        load = list(dict.fromkeys([
            *cfg.f0.dataset_load,
            *(cfg.codes.dataset_load if "codes" in cfg else []),
        ]))
        # The pitch-to-code classifier needs only cached tensors when both
        # supervision sources are cached. Preserve audio for the F0 probe.
        if "codes" in cfg and cfg.codes.source == "cached" and not cfg.f0.online_encoders:
            load = [name for name in load if name != "audio"]
        return instantiate(
            cfg.dataset, splits=[split], load=load,
            resource_providers=providers, target_sr=16000,
        )

    train_dataset = dataset(cfg.train_split)
    val_dataset = dataset(cfg.val_split) if cfg.get("val_split") is not None else None
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


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    run_training(cfg)


if __name__ == "__main__":
    main()
