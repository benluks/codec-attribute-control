"""Precompute tiny FACodec prosody-code tensors with QC's existing pipeline."""
import hydra
from quick_convert.cli.run import execute_pipeline
from . import train as _train  # noqa: F401


@hydra.main(version_base=None, config_path="configs", config_name="precompute_codes")
def main(cfg):
    execute_pipeline(cfg)


if __name__ == "__main__":
    main()
