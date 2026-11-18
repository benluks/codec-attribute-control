"""Compare online JDC batching with the single-utterance path."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from codec_attribute_control.f0 import aligned_pitch
from codec_attribute_control.f0_resource import OnlineF0
from codec_attribute_control.facodec import HOP, SAMPLE_RATE


@pytest.fixture(
    scope="module", params=[(16000, 16000), (12037, 16000)], ids=["equal", "mixed"]
)
def jdc_outputs(request):
    online = OnlineF0()
    extractor = online.extractor
    lengths = torch.tensor(request.param)
    recordings = [
        0.2 * torch.sin(torch.arange(n) * (2 * torch.pi * hz / SAMPLE_RATE))
        for n, hz in zip(request.param, (173, 217))
    ]
    batch = SimpleNamespace(
        waveforms=pad_sequence(recordings, batch_first=True),
        lengths=lengths,
        sample_rates=torch.full_like(lengths, SAMPLE_RATE),
    )
    positions = torch.arange(batch.waveforms.shape[1])
    batch.waveforms.masked_fill_(positions >= lengths[:, None], 0.5)
    batch_mels = []
    model_inputs = []

    def capture_batch_mel(waveforms, sizes):
        mel = batch_log_mel(waveforms, sizes)
        batch_mels.append(mel[:, None])
        return mel

    batch_log_mel = extractor._batch_log_mel
    with patch.object(extractor, "_batch_log_mel", side_effect=capture_batch_mel):
        batched_pitch = online.train()(batch)
    (batched_mel,) = batch_mels
    assert not extractor.model.training

    def capture_mel(_model, args):
        model_inputs.append(args[0].detach())

    hook = extractor.model.register_forward_pre_hook(capture_mel)
    try:
        frame_lengths = (lengths + HOP - 1) // HOP
        single_pitch = [
            torch.from_numpy(aligned_pitch(extractor, recording, SAMPLE_RATE, int(n)))
            for recording, n in zip(recordings, frame_lengths)
        ]
        single_mels = model_inputs
    finally:
        hook.remove()
    assert torch.equal(batched_pitch.lengths, frame_lengths)
    return batched_mel, single_mels, batched_pitch.values, single_pitch


def test_jdc_batched_mel_matches_single(jdc_outputs):
    batched_mel, single_mels, _, _ = jdc_outputs
    for i, single in enumerate(single_mels):
        torch.testing.assert_close(
            batched_mel[i : i + 1, ..., : single.shape[-1]],
            single,
            rtol=1e-4,
            atol=1e-5,
        )


def test_jdc_batched_pitch_matches_single(jdc_outputs):
    _, _, batched_pitch, single_pitch = jdc_outputs
    for i, single in enumerate(single_pitch):
        torch.testing.assert_close(
            batched_pitch[i, : single.numel(), 0],
            single,
            rtol=1e-4,
            atol=1e-3,
            equal_nan=True,
        )
        assert torch.isnan(batched_pitch[i, single.numel() :]).all()
