"""Batched FACodec adapter contract without pretrained downloads."""
from types import SimpleNamespace

import torch

from codec_attribute_control.facodec_encoder import FACodecContentEncoder


class BatchCodec:
    def __init__(self):
        self.calls = 0

    def encode_batch(self, waveforms, lengths):
        self.calls += 1
        b, n = waveforms.shape
        t = (n + 199) // 200
        def stream(v):
            return torch.full((b, 256, t), v)
        return SimpleNamespace(
            combined=stream(6), prosody=stream(1), content=stream(2),
            residual=stream(3), speaker=torch.ones(b, 256),
            codes=torch.zeros(6, b, t),
        )


def test_batched_path_runs_once_and_masks_padding():
    codec = BatchCodec()
    encoder = FACodecContentEncoder(codec=codec)
    result = encoder.encode_waveforms(torch.randn(3, 800), torch.tensor([800, 400, 600]), 16000)
    assert codec.calls == 1
    assert result.lengths.tolist() == [4, 2, 3]
    assert result.prosody.shape == (3, 4, 256)
    assert torch.all(result.prosody[1, 2:] == 0)
    assert torch.all(result.prosody[2, 3:] == 0)
    assert torch.all(result.prosody[0] == 1)
