"""Use Quick Convert's provenance-aware feature precompute pipeline."""
import hydra
from quick_convert.cli.run import execute_pipeline
from . import train as _train  # noqa: F401

@hydra.main(version_base=None, config_path="configs", config_name="precompute_f0")
def main(cfg):
    execute_pipeline(cfg)

if __name__ == "__main__":
    main()
