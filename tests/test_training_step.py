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
