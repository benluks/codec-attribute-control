import pytest
import torch

from codec_attribute_control.latent_inversion import invert_pitch, shift_log2_f0


def test_inversion_moves_pitch_without_updating_probe():
    probe = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        probe.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))
    source = torch.zeros(1, 8, 3)
    target = torch.full((1, 8), 0.5)
    weights = probe.weight.detach().clone()
    result = invert_pitch(
        lambda_probe(probe), source, target,
        steps=120, learning_rate=0.05, displacement_weight=0.001,
    )
    assert result.pitch_loss < 0.002
    assert torch.allclose(probe.weight, weights)
    assert torch.allclose(source, torch.zeros_like(source))


class lambda_probe(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, x):
        return self.layer(x).squeeze(-1)


def test_mask_and_nan():
    probe = lambda_probe(torch.nn.Linear(2, 1, bias=False))
    x = torch.zeros(1, 3, 2)
    y = torch.tensor([[0.5, float("nan"), 0.5]])
    result = invert_pitch(probe, x, y, steps=2)
    assert torch.isfinite(result.latents).all()
    with pytest.raises(ValueError, match="No valid"):
        invert_pitch(probe, x, y, valid_mask=torch.zeros(1, 3, dtype=torch.bool))


def test_semitone_shift_preserves_nan():
    x = torch.tensor([[5.0, float("nan")]])
    shifted = shift_log2_f0(x, 12)
    assert shifted[0, 0] == 6.0
    assert torch.isnan(shifted[0, 1])
