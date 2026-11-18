"""Run pretrained parity checks with FACODEC_PARITY=1 uv run --extra train --extra facodec pytest."""

import os

import pytest
import torch

from codec_attribute_control.facodec import HOP, FACodec
from codec_attribute_control.facodec_encoder import FACodecContentEncoder

pytestmark = pytest.mark.skipif(
    os.environ.get("FACODEC_PARITY") != "1",
    reason="Requires pretrained FACodec weights; set FACODEC_PARITY=1",
)


@pytest.fixture(scope="module")
def codec():
    return FACodec(device="cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "lengths, extra_padding",
    [
        ([16037], 37),
        ([12037, 12000 + HOP, 12091], 0),
        ([12037, 24000, 8000, 15091, 1], 37),
        ([24000, 12037], 0),
    ],
)
def test_pretrained_batch_parity(codec, lengths, extra_padding, monkeypatch):
    if len(lengths) == 5:
        monkeypatch.setattr("codec_attribute_control.facodec.MAX_BATCH_PADDED_SAMPLES", 36000)
    generator = torch.Generator().manual_seed(20261118)
    audio = [torch.randn(n, generator=generator) * 0.08 for n in lengths]
    # Leave garbage after each true end, including extra collator padding.
    waveforms = torch.randn(len(lengths), max(lengths) + extra_padding, generator=generator)
    for row, recording in zip(waveforms, audio):
        row[: recording.numel()] = recording

    adapter = FACodecContentEncoder(codec=codec)
    batched = adapter.train().encode_waveforms(waveforms, torch.tensor(lengths))
    assert not codec.encoder.training
    assert not codec.decoder.training

    for i, recording in enumerate(audio):
        ref = codec.encode(recording)
        frames = (lengths[i] + HOP - 1) // HOP
        assert batched.lengths[i].item() == frames
        for name in ("combined", "prosody", "content", "residual"):
            stream = batched.values if name == "combined" else batched.streams[name]
            torch.testing.assert_close(
                stream[i : i + 1, :frames].transpose(1, 2),
                getattr(ref, name),
                rtol=1e-4,
                atol=1e-5,
            )
            assert torch.count_nonzero(stream[i, frames:]) == 0
        assert torch.equal(batched.codes[:, i : i + 1, :frames], ref.codes)
        assert torch.count_nonzero(batched.codes[:, i, frames:]) == 0
        torch.testing.assert_close(
            batched.speaker[i : i + 1], ref.speaker, rtol=1e-4, atol=1e-5
        )
