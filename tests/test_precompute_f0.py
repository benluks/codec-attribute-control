"""Cache writer tests using QC's batch contract; no JDC weights required."""
from pathlib import Path

import torch
from quick_convert.data.types import AudioBatch, AudioSample
from quick_convert.components.resources.resolution import ResolvedResource

from codec_attribute_control.precompute_f0 import precompute


class Dataset:
    def __init__(self):
        self.samples = [
            AudioSample(utt_id="a", path=Path("a.flac"), split="train-clean-100",
                        waveform=torch.ones(1, 400), sample_rate=16000),
            AudioSample(utt_id="b", path=Path("b.flac"), split="train-clean-100",
                        waveform=torch.ones(1, 200), sample_rate=16000),
        ]

    def make_dataloader(self, **kwargs):
        return [AudioBatch.from_samples(self.samples)]


class Extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, batch):
        self.calls += 1
        lengths = (batch.lengths + 199) // 200
        return ResolvedResource(
            values=torch.ones(len(lengths), int(lengths.max()), 1),
            lengths=lengths,
        )


def test_precompute_writes_qc_compatible_tensors_and_skips(tmp_path):
    dataset, extractor = Dataset(), Extractor()
    assert precompute(dataset, extractor, tmp_path) == (2, 0)
    assert torch.load(tmp_path / "train-clean-100/a.pt", weights_only=True).shape == (2, 1)
    assert torch.load(tmp_path / "train-clean-100/b.pt", weights_only=True).shape == (1, 1)
    assert precompute(dataset, extractor, tmp_path) == (0, 2)
    assert extractor.calls == 1
    assert precompute(dataset, extractor, tmp_path, overwrite=True) == (2, 0)
    assert extractor.calls == 2
