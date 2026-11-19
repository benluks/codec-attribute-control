"""Synthetic QC-style F0 resource training without model downloads."""
from types import SimpleNamespace

import torch
from torch import nn

from codec_attribute_control.head import LinearF0Head
from codec_attribute_control.system import F0ProbeSystem
from codec_attribute_control.training import F0ProbeTrainingModule


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 4)

    def forward(self, batch):
        features = self.layer(batch.waveforms)
        return SimpleNamespace(values=features, streams={"prosody": features},
                               lengths=torch.tensor([4, 3], device=features.device))


class DummyF0(nn.Module):
    def forward(self, batch):
        return batch.test_f0, torch.tensor([4, 3])


def test_training_step_online_and_cached():
    for cached in (False, True):
        encoder = DummyEncoder()
        system = F0ProbeSystem(encoder, LinearF0Head(4),
                               online_encoders={} if cached else {"f0": DummyF0()})
        module = F0ProbeTrainingModule(system=system, optimization=None)
        module.log = lambda *args, **kwargs: None
        f0 = torch.tensor([[[100.], [110.], [120.], [130.]],
                           [[200.], [220.], [240.], [0.]]])
        batch = SimpleNamespace(
            waveforms=torch.randn(2, 4, 2),
            test_f0=f0,
            resources={"f0": SimpleNamespace(values=f0, lengths=torch.tensor([4, 3]))}
            if cached else {},
        )
        loss = module._shared_step(batch, "train").loss
        assert torch.isfinite(loss)
        loss.backward()
        assert system.head.projection.weight.grad is not None
        assert encoder.layer.weight.grad is None


def test_contours_log_with_lightning_step_metric(monkeypatch):
    from unittest.mock import Mock

    import matplotlib

    matplotlib.use("Agg")
    import wandb
    from lightning.pytorch.loggers import WandbLogger

    logger = WandbLogger(project="test-contour-logging")
    experiment = Mock()
    monkeypatch.setattr(logger, "_experiment", experiment)
    module = F0ProbeTrainingModule(system=nn.Identity(), optimization=None)
    module._trainer = SimpleNamespace(logger=logger, global_step=3568)
    f0_hz = torch.tensor([[100.0, 110.0, 120.0, 130.0]])
    module._log_validation_curves(f0_hz, torch.zeros_like(f0_hz), torch.tensor([4]))

    experiment.log.assert_called_once()
    args, kwargs = experiment.log.call_args
    assert kwargs == {}
    assert args[0]["trainer/global_step"] == 3568
    (image,) = args[0]["val/f0_curves_hz"]
    assert isinstance(image, wandb.Image)
