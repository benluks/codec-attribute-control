"""Hydra training entry point following speech-semantic-vectors."""
from pathlib import Path

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import OmegaConf
from quick_convert.data import BaseDataset
from quick_convert.pipelines.training.pipeline import TrainingPipeline
from quick_convert.utils.resolvers import register_config_resolvers

register_config_resolvers()


@hydra.main(version_base=None, config_path="configs", config_name="train")
def main(cfg):
    dataset = instantiate(cfg.dataset)
    if len(dataset) < 2:
        raise ValueError("At least two recordings are required for recording-disjoint validation")
    rng = np.random.default_rng(cfg.seed)
    indices = rng.permutation(len(dataset))
    n_val = max(1, round(len(indices) * cfg.val_fraction))
    train_rows = [dataset.rows[int(i)] for i in indices[n_val:]]
    val_rows = [dataset.rows[int(i)] for i in indices[:n_val]]
    # Preserve QC's loading and collation rather than implementing a second dataloader.
    common = dict(load=["audio"], target_sr=dataset.target_sr,
                  convert_to_mono=dataset.convert_to_mono)
    train_dataset = BaseDataset(rows=train_rows, **common)
    val_dataset = BaseDataset(rows=val_rows, **common)
    trainer = instantiate(cfg.trainer)
    pipeline = TrainingPipeline(
        trainer=trainer, train_dataset=train_dataset, val_dataset=val_dataset,
        out_dir=Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir),
    )
    pipeline.prepare()
    pipeline.write_config(OmegaConf.to_yaml(cfg, resolve=True))
    pipeline.log_config(OmegaConf.to_container(cfg, resolve=True))
    pipeline.run()


if __name__ == "__main__":
    main()
