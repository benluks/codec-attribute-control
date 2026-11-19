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

    def forward(self, f0_hz, lengths=None):
        if f0_hz.ndim != 3 or f0_hz.shape[-1] != 2:
            raise ValueError("Expected pitch features [B,T,2]")
        x = f0_hz
        if self.context_size > 1:
            k = self.context_size // 2
            x = F.pad(x.transpose(1, 2), (k, k), mode="replicate")
            x = x.unfold(-1, self.context_size, 1).permute(0, 2, 1, 3).flatten(2)
        return self.network(x)


class ResidualDilatedBlock(nn.Module):
    """Noncausal residual temporal convolution."""

    def __init__(self, channels, dilation):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm = nn.LayerNorm(channels)
        self.activation = nn.GELU()

    def forward(self, x):
        y = self.conv(x).transpose(1, 2)
        y = self.activation(self.norm(y)).transpose(1, 2)
        return x + y


class GlobalContextPitchCodeHead(nn.Module):
    """Dilated CNN with masked utterance-wide statistics for code classification."""

    def __init__(self, codebook_size=1024, hidden_size=128, dilations=(1, 2, 4, 8, 16, 32), normalization='none', global_mean=None, global_std=None):
        super().__init__()
        self.input_projection = nn.Conv1d(2, hidden_size, 1)
        self.blocks = nn.ModuleList([ResidualDilatedBlock(hidden_size, d) for d in dilations])
        self.output_projection = nn.Sequential(
            nn.Conv1d(3 * hidden_size, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, codebook_size, 1),
        )

    def forward(self, f0_hz, lengths=None):
        if f0_hz.ndim != 3 or f0_hz.shape[-1] != 2:
            raise ValueError("Expected pitch features [B,T,2]")
        b, t, _ = f0_hz.shape
        if lengths is None:
            lengths = torch.full((b,), t, device=f0_hz.device, dtype=torch.long)
        if lengths.shape != (b,) or torch.any(lengths < 1) or torch.any(lengths > t):
            raise ValueError("Invalid lengths")
        mask = torch.arange(t, device=f0_hz.device)[None, :] < lengths[:, None]
        x = self.input_projection(f0_hz.transpose(1, 2))
        m = mask.unsqueeze(1)
        for block in self.blocks:
            x = block(x * m) * m
        count = lengths.to(x.dtype).view(b, 1, 1)
        mean = x.sum(dim=-1, keepdim=True) / count
        variance = ((x - mean) * m).square().sum(dim=-1, keepdim=True) / count
        std = (variance + 1e-6).sqrt()
        context = torch.cat([x, mean.expand_as(x), std.expand_as(x)], dim=1)
        return self.output_projection(context).transpose(1, 2)
