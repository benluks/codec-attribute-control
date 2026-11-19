import pytest
import torch
from codec_attribute_control.pitch_transform import pitch_inputs, PitchInputTransform
from codec_attribute_control.code_head import GlobalContextPitchCodeHead


def test_utterance_normalization_ignores_unvoiced_and_padding():
    f0 = torch.tensor([[100., 200., 0., 400., 1000.]])
    pitch, voiced, valid = pitch_inputs(f0, torch.tensor([4]), "utterance")
    assert voiced.tolist() == [[True, True, False, True, False]]
    assert valid.tolist() == [[True, True, True, True, False]]
    assert pitch[0, 2] == 0 and pitch[0, 4] == 0
    assert torch.isclose(pitch[0, voiced[0]].mean(), torch.tensor(0.), atol=1e-5)
    assert torch.isclose(pitch[0, voiced[0]].square().sum(), torch.tensor(2.), atol=1e-5)


def test_constant_and_unvoiced_are_finite():
    x = torch.tensor([[100., 100., 0.], [0., 0., 0.]])
    y, _, _ = pitch_inputs(x, normalization="utterance")
    assert torch.isfinite(y).all()
    assert (y == 0).all()


def test_global_requires_stats():
    with pytest.raises(ValueError):
        pitch_inputs(torch.tensor([[100.]]), normalization="global")
    y, _, _ = pitch_inputs(torch.tensor([[128., 0.]]), normalization="global", global_mean=7., global_std=2.)
    assert y.tolist() == [[0., 0.]]


@pytest.mark.parametrize("normalization", ["none", "utterance", "global"])
def test_cnn_shape(normalization):
    kwargs = {"global_mean": 7., "global_std": 2.} if normalization == "global" else {}
    head = GlobalContextPitchCodeHead(codebook_size=12, hidden_size=8, dilations=(1,))
    transformed = PitchInputTransform(normalization=normalization, **kwargs)(torch.tensor([[100., 200., 0.]]))
    logits = head(transformed, lengths=torch.tensor([3]))
    assert logits.shape == (1, 3, 12)
