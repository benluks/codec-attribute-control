"""QC ContentEncoder exposing FACodec's factorized post-VQ representation."""
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence

from quick_convert.components.encoders.content.base import ContentEncoder, ContentFeatures
from .facodec import FACodec, HOP, SAMPLE_RATE


@dataclass
class FACodecFeatures(ContentFeatures):
    """Combined latent in .values; named components retained for reconstruction."""
    streams: dict[str, torch.Tensor] = None
    speaker: torch.Tensor | None = None
    codes: torch.Tensor | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.streams is None or set(self.streams) != {"prosody", "content", "residual"}:
            raise ValueError("FACodec requires prosody, content and residual streams")
        for name, value in self.streams.items():
            if value.shape != self.values.shape:
                raise ValueError(f"{name} must have shape {tuple(self.values.shape)}")
        if not torch.allclose(self.values, sum(self.streams.values()), rtol=1e-4, atol=1e-4):
            raise ValueError("FACodec combined latent must equal the sum of streams")
        if self.speaker is None or self.codes is None:
            raise ValueError("FACodec reconstruction requires speaker and codes")

    @property
    def prosody(self):
        return self.streams["prosody"]


class FACodecContentEncoder(ContentEncoder):
    """Adapter for the frozen upstream codec; all streams survive encoding.

    Currently invokes the upstream single-recording API per item. This is
    intentionally not advertised as vectorized FACodec execution.
    """
    FEATURE_DIM = 256

    def __init__(self, device="cpu", codec=None):
        super().__init__(device=device)
        self.codec = codec if codec is not None else FACodec(device=str(self.device))

    @property
    def sample_rate(self):
        return SAMPLE_RATE

    @property
    def frame_hz(self):
        return SAMPLE_RATE / HOP

    def encode_waveforms(self, waveforms, lengths=None, sample_rate=None):
        if sample_rate is not None and sample_rate != SAMPLE_RATE:
            raise ValueError(f"Expected {SAMPLE_RATE} Hz audio")
        if waveforms.ndim != 2:
            raise ValueError("Expected [B, samples] waveforms")
        b, n = waveforms.shape
        if lengths is None:
            lengths = torch.full((b,), n, dtype=torch.long, device=waveforms.device)
        if lengths.shape != (b,) or torch.any(lengths <= 0) or torch.any(lengths > n):
            raise ValueError("Invalid waveform lengths")
        frames = [self.codec.encode(waveforms[i, :int(lengths[i])]) for i in range(b)]
        def padded(field):
            return pad_sequence(
                [getattr(f, field)[0].transpose(0, 1).clone() for f in frames],
                batch_first=True,
            )
        combined = padded("combined")
        streams = {name: padded(name) for name in ("prosody", "content", "residual")}
        frame_lengths = torch.tensor([f.combined.shape[-1] for f in frames],
                                     dtype=torch.long, device=combined.device)
        return FACodecFeatures(
            values=combined, lengths=frame_lengths, feature_dim=self.FEATURE_DIM,
            representation_type="continuous", temporal_granularity="frame",
            backend="facodec", model_name="amphion/naturalspeech3_facodec",
            layer=None, frame_hz=self.frame_hz, streams=streams,
            speaker=torch.cat([f.speaker for f in frames], dim=0),
            codes=pad_sequence(
                [f.codes.permute(2, 0, 1) for f in frames], batch_first=True
            ).permute(0, 2, 3, 1),
        )

    def forward(self, batch):
        if batch.waveforms is None or batch.lengths is None:
            raise ValueError("FACodec requires loaded audio")
        if not torch.all(batch.sample_rates == SAMPLE_RATE):
            raise ValueError(f"FACodec requires {SAMPLE_RATE} Hz audio")
        return self.encode_waveforms(batch.waveforms, batch.lengths, SAMPLE_RATE)

    def encode_file(self, path: str | Path):
        waveform, sr = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
        return self.encode_waveforms(waveform, sample_rate=SAMPLE_RATE)
