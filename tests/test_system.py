"""Test system/head composition without downloading FACodec checkpoints."""
import torch
from torch import nn
from types import SimpleNamespace

from codec_attribute_control.head import LinearF0Head
from codec_attribute_control.system import F0ProbeSystem


class DummyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 4)

    def forward(self, batch):
        return SimpleNamespace(
            values=self.projection(batch),
            lengths=torch.tensor([3, 2], device=batch.device),
        )


def test_batched_probe_and_freeze():
    encoder = DummyEncoder()
    system = F0ProbeSystem(encoder=encoder, head=LinearF0Head(input_dim=4))
    predictions, lengths = system(torch.randn(2, 3, 2))
    assert predictions.shape == (2, 3)
    assert lengths.tolist() == [3, 2]
    predictions.sum().backward()
    assert system.head.projection.weight.grad is not None
    assert encoder.projection.weight.grad is None


def test_head_masked_loss():
    head = LinearF0Head(input_dim=4)
    predictions = torch.tensor([[1., 100., 3.]])
    targets = torch.tensor([[0., 0., 1.]])
    valid = torch.tensor([[True, False, True]])
    assert torch.isclose(head.loss(predictions, targets, valid), torch.tensor(2.5))
