"""Contract tests for the QC-compatible probe module."""
import torch
from quick_convert.training.lightning.modules.base import BaseTrainingModule

from codec_attribute_control.training import F0ProbeTrainingModule


def test_probe_inherits_qc_training_module():
    assert issubclass(F0ProbeTrainingModule, BaseTrainingModule)


def test_probe_step_is_differentiable():
    module = F0ProbeTrainingModule(feature_dim=4, optimization=None)
    module.log = lambda *args, **kwargs: None
    batch = {
        "features": torch.randn(2, 5, 4),
        "targets": torch.randn(2, 5),
        "voiced": torch.tensor([[True, True, False, True, False],
                                 [True, False, True, True, True]]),
    }
    result = module._shared_step(batch, "train")
    assert result.loss.ndim == 0
    result.loss.backward()
    assert module.probe.weight.grad is not None


def test_unvoiced_batch_has_connected_zero_loss():
    module = F0ProbeTrainingModule(feature_dim=4, optimization=None)
    module.log = lambda *args, **kwargs: None
    batch = {"features": torch.randn(1, 3, 4),
             "targets": torch.full((1, 3), -10.0),
             "voiced": torch.zeros((1, 3), dtype=torch.bool)}
    loss = module._shared_step(batch, "train").loss
    loss.backward()
    assert loss.item() == 0
