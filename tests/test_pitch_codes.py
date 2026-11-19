from codec_attribute_control.pitch_transform import PitchInputTransform
import pytest
import torch
from codec_attribute_control.code_head import PitchCodeHead


@pytest.mark.parametrize("window", [1, 3, 5])
def test_pitch_code_head_shape_gradients_and_unvoiced(window):
    head = PitchCodeHead(codebook_size=16, hidden_sizes=[8], context_size=window)
    f0 = torch.tensor([[100., float("nan"), 200., 0., 150.]])
    logits = head(PitchInputTransform()(f0))
    assert logits.shape == (1, 5, 16)
    assert torch.isfinite(logits).all()
    logits.sum().backward()
    assert all(p.grad is not None for p in head.parameters())


def test_pitch_code_head_rejects_even_window():
    with pytest.raises(ValueError, match="context_size"):
        PitchCodeHead(context_size=4)
