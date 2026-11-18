"""Online F0 resource on the FACodec frame grid."""
import torch
import torchaudio.functional as audio_functional
from quick_convert.components.resources.resolution import ResolvedResource
from torch import nn

from .extractors import JDC_SAMPLE_RATE, JDCF0Extractor
from .f0 import aligned_pitch, make_extractor
from .facodec import HOP, SAMPLE_RATE


class OnlineF0(nn.Module):
    """QC-compatible online resource encoder; returns Hz [B,T,1] and lengths."""

    def __init__(self, backend="jdc", device="cpu", checkpoint_path=None):
        super().__init__()
        self.extractor = make_extractor(backend, device=device, checkpoint_path=checkpoint_path)

    @torch.no_grad()
    def forward(self, batch):
        if batch.waveforms is None or batch.lengths is None:
            raise ValueError("Online F0 requires loaded audio")
        if not torch.all(batch.sample_rates == SAMPLE_RATE):
            raise ValueError("Online F0 requires 16 kHz audio")
        if isinstance(self.extractor, JDCF0Extractor):
            device = batch.waveforms.device
            lengths = batch.lengths.to(device=device, dtype=torch.long)
            # Resample the entire padded batch in one operation.
            # Native valid lengths use torchaudio's ceil(input * new / old).
            positions = torch.arange(batch.waveforms.shape[1], device=device)
            waveforms = batch.waveforms.masked_fill(positions >= lengths[:, None], 0)
            padded = audio_functional.resample(waveforms, SAMPLE_RATE, JDC_SAMPLE_RATE)
            native_lengths = torch.div(
                lengths * JDC_SAMPLE_RATE + SAMPLE_RATE - 1,
                SAMPLE_RATE,
                rounding_mode="floor",
            )
            raw, raw_lengths = self.extractor.extract_batch(padded, native_lengths, JDC_SAMPLE_RATE)
            frame_lengths = (lengths + HOP - 1) // HOP
            positions = torch.arange(int(frame_lengths.max()), device=raw.device)
            # Both JDC and FACodec use 12.5 ms frames.
            index = positions.clamp(max=raw.shape[-1] - 1)
            pitch = raw[:, index]
            pitch = pitch.masked_fill(
                (positions[None, :] >= raw_lengths[:, None]) |
                (positions[None, :] >= frame_lengths.to(raw.device)[:, None]) |
                ~torch.isfinite(pitch) | (pitch <= 5.0),
                float("nan"),
            )
            return ResolvedResource(values=pitch.unsqueeze(-1), lengths=frame_lengths)
        contours = []
        for i, n in enumerate(batch.lengths.tolist()):
            n_frames = (n + HOP - 1) // HOP
            pitch = aligned_pitch(self.extractor, batch.waveforms[i, :n].detach().cpu(),
                                  SAMPLE_RATE, n_frames)
            contours.append(torch.as_tensor(pitch.copy(), dtype=torch.float32,
                                            device=batch.waveforms.device)[:, None])
        lengths = torch.tensor([x.shape[0] for x in contours],
                               dtype=torch.long, device=batch.waveforms.device)
        return ResolvedResource(values=torch.nn.utils.rnn.pad_sequence(contours, batch_first=True),
                                lengths=lengths)
