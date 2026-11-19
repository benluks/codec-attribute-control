"""Use Quick Convert's provenance-aware feature precompute pipeline."""

import hydra
from omegaconf import open_dict
from quick_convert.cli.run import execute_pipeline

from . import train as _train  # noqa: F401


@hydra.main(version_base=None, config_path="configs", config_name="precompute_f0")
def main(cfg):
    if cfg.f0_backend == "jdc":
        # QC compares the resolved extractor configuration when resuming a cache.
        with open_dict(cfg.pipeline.extractor):
            cfg.pipeline.extractor._target_ = "codec_attribute_control.f0_precompute_extractor.JDCF0PrecomputeExtractor"
            del cfg.pipeline.extractor.backend
    execute_pipeline(cfg)


if __name__ == "__main__":
    main()
