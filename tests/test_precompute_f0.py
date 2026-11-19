"""Adapter contract without pretrained weights."""

from pathlib import Path
from types import SimpleNamespace

import torch
from quick_convert.data.types import AudioBatch, AudioSample

from codec_attribute_control.f0_precompute_extractor import JDCF0PrecomputeExtractor


def test_adapter_returns_unpadded_contours():
    adapter = JDCF0PrecomputeExtractor.__new__(JDCF0PrecomputeExtractor)
    torch.nn.Module.__init__(adapter)

    class FakeOnline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.extractor = torch.nn.Module()
            self.extractor.model = torch.nn.Linear(1, 1)

        def forward(self, batch):
            return SimpleNamespace(
                values=torch.ones(2, 3, 1), lengths=torch.tensor([3, 1])
            )

    adapter.online = FakeOnline()
    samples = [
        AudioSample(
            utt_id=str(i),
            path=Path(str(i)),
            split="test-clean",
            waveform=torch.ones(1, 400 - i * 200),
            sample_rate=16000,
        )
        for i in range(2)
    ]
    output = adapter.extract_batch(AudioBatch.from_samples(samples))
    assert [x.shape for x in output] == [(3, 1), (1, 1)]


def test_world_cache_retains_pitch_and_voicing():
    import pytest

    from codec_attribute_control.f0_precompute_extractor import F0PrecomputeExtractor

    pytest.importorskip("pyworld")
    t = torch.arange(16000) / 16000
    tone = 0.2 * sum(torch.sin(2 * torch.pi * 200 * k * t) / k for k in range(1, 7))
    audio = torch.cat([torch.zeros(4000), tone, torch.zeros(4001)])
    batch = AudioBatch.from_samples(
        [
            AudioSample(
                utt_id="tone",
                path=Path("tone"),
                split="test-clean",
                waveform=audio[None],
                sample_rate=16000,
            ),
            AudioSample(
                utt_id="silence",
                path=Path("silence"),
                split="test-clean",
                waveform=torch.zeros(1, 4001),
                sample_rate=16000,
            ),
        ]
    )
    extractor = F0PrecomputeExtractor(backend="world", device="cuda")
    pitch, silence = extractor.extract_batch(batch)
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        extractor.pitch_pool = pool
        parallel = extractor.extract_batch(batch)
        repeated = extractor.extract_batch(batch)
    for expected, first, second in zip(
        (pitch, silence), parallel, repeated, strict=True
    ):
        torch.testing.assert_close(first, expected, atol=0, rtol=0, equal_nan=True)
        torch.testing.assert_close(second, expected, atol=0, rtol=0, equal_nan=True)
    assert pitch.shape == (121, 1)
    assert silence.shape == (21, 1)
    assert torch.isnan(pitch[:10]).all() and torch.isnan(pitch[-10:]).all()
    assert torch.isnan(silence).all()
    torch.testing.assert_close(pitch[30:85], torch.full((55, 1), 200.0), atol=2, rtol=0)
