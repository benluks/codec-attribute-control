"""Pitch-to-discrete-prosody-code classifier."""
import itertools
import torch
from torch import nn
import torch.nn.functional as F


class PitchCodeHead(nn.Module):
    """Centered temporal MLP predicting one FACodec prosody code per frame."""

    def __init__(self, codebook_size=1024, hidden_sizes=(128,), context_size=5):
        super().__init__()
        if type(context_size) is not int or context_size < 1 or context_size % 2 == 0:
            raise ValueError("context_size must be a positive odd integer")
        self.context_size = context_size
        dims = [2 * context_size, *hidden_sizes, codebook_size]
        layers = []
        for a, b in itertools.pairwise(dims[:-1]):
            layers.extend([nn.Linear(a, b), nn.GELU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, f0_hz):
        if f0_hz.ndim != 2:
            raise ValueError("Expected F0 [B,T] in Hz")
        voiced = torch.isfinite(f0_hz) & (f0_hz > 5)
        log_pitch = torch.where(voiced, f0_hz.clamp_min(5).log2(), 0.)
        x = torch.stack([log_pitch, voiced.to(log_pitch.dtype)], dim=-1)
        if self.context_size > 1:
            k = self.context_size // 2
            x = F.pad(x.transpose(1, 2), (k, k), mode="replicate")
            x = x.unfold(-1, self.context_size, 1).permute(0, 2, 1, 3).flatten(2)
        return self.network(x)
