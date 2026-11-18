"""Training-module structure checks (no pretrained model download)."""
from quick_convert.training.lightning.modules.base import BaseTrainingModule
from codec_attribute_control.training import F0ProbeTrainingModule


def test_probe_inherits_qc_training_module():
    assert issubclass(F0ProbeTrainingModule, BaseTrainingModule)
