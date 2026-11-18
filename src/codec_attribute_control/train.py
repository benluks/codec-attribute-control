"""Hydra entry point; QC owns dataset loading and training."""
from importlib.resources import files
from pathlib import Path

import hydra
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin
from hydra.utils import instantiate
from omegaconf import OmegaConf
from quick_convert.pipelines.training.pipeline import TrainingPipeline
from quick_convert.utils.resolvers import register_config_resolvers

class QuickConvertConfigSearchPath(SearchPathPlugin):
    def manipulate_search_path(self, search_path):
        config_dir = files("quick_convert").joinpath("configs")
        search_path.append(provider="quick_convert", path=f"file://{config_dir}")


Plugins.instance().register(QuickConvertConfigSearchPath)
register_config_resolvers()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    providers = instantiate(cfg.f0.resource_providers)
    def dataset(split):
        return instantiate(cfg.dataset, splits=[split], load=cfg.f0.dataset_load,
                           resource_providers=providers, target_sr=16000)

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


if __name__ == "__main__":
    main()
