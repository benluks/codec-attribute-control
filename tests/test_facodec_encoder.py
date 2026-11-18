"""Contract tests without FACodec weights."""
from types import SimpleNamespace

import pytest
import torch

from codec_attribute_control.facodec_encoder import FACodecContentEncoder


class DummyCodec:
    def encode(self, waveform):
        t = (waveform.numel() + 199) // 200
        def stream(value):
            return torch.full((1, 256, t), value)
        return SimpleNamespace(
            combined=stream(6.), prosody=stream(1.),
            content=stream(2.), residual=stream(3.),
            speaker=torch.ones(1, 256), codes=torch.zeros(6, 1, t),
        )


def test_factorized_features():
    encoder = FACodecContentEncoder(codec=DummyCodec())
    features = encoder.encode_waveforms(
        torch.randn(2, 600), torch.tensor([600, 400]), 16000
    )
    assert features.values.shape == (2, 3, 256)
    assert features.prosody.shape == features.values.shape
    assert features.lengths.tolist() == [3, 2]
    assert torch.all(features.prosody[0, :3] == 1)
    assert torch.all(features.prosody[1, :2] == 1)
    assert torch.all(features.prosody[1, 2:] == 0)


def test_sample_rate_validation():
    encoder = FACodecContentEncoder(codec=DummyCodec())
    with pytest.raises(ValueError):
        encoder.encode_waveforms(torch.randn(1, 600), sample_rate=24000)
