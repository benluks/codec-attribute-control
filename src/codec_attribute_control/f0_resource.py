"""Online F0 resource on the FACodec frame grid."""
import torch
from torch import nn

from .f0 import aligned_pitch, make_extractor
from .facodec import HOP, SAMPLE_RATE


class OnlineF0(nn.Module):
    """QC-compatible online resource encoder; returns Hz [B,T,1] and lengths."""

    def __init__(self, backend="yaapt", device="cpu", checkpoint_path=None):
        super().__init__()
        self.extractor = make_extractor(backend, device=device, checkpoint_path=checkpoint_path)

    @torch.no_grad()
    def forward(self, batch):
        if batch.waveforms is None or batch.lengths is None:
            raise ValueError("Online F0 requires loaded audio")
        if not torch.all(batch.sample_rates == SAMPLE_RATE):
            raise ValueError("Online F0 requires 16 kHz audio")
        contours = []
        for i, n in enumerate(batch.lengths.tolist()):
            n_frames = (n + HOP - 1) // HOP
            pitch = aligned_pitch(self.extractor, batch.waveforms[i, :n].detach().cpu(),
                                  SAMPLE_RATE, n_frames)
            contours.append(torch.as_tensor(pitch.copy(), dtype=torch.float32,
                                            device=batch.waveforms.device)[:, None])
        lengths = torch.tensor([x.shape[0] for x in contours],
                               dtype=torch.long, device=batch.waveforms.device)
        return torch.nn.utils.rnn.pad_sequence(contours, batch_first=True), lengths
