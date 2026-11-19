"""Pitch-to-discrete-prosody-code classifier."""
import itertools
import torch
from torch import nn
import torch.nn.functional as F


def pitch_inputs(f0_hz, lengths=None, normalization="none", global_mean=None, global_std=None):
    """Log2 pitch plus voicing; normalize only voiced, valid frames."""
    if normalization not in ("none", "utterance", "global"):
        raise ValueError(f"Unsupported pitch normalization: {normalization}")
    if f0_hz.ndim != 2:
        raise ValueError("Expected F0 [B,T] in Hz")
    b, t = f0_hz.shape
    if lengths is None:
        lengths = torch.full((b,), t, device=f0_hz.device, dtype=torch.long)
    if lengths.shape != (b,) or torch.any(lengths < 1) or torch.any(lengths > t):
        raise ValueError("Invalid F0 sequence lengths")
    valid = torch.arange(t, device=f0_hz.device)[None, :] < lengths[:, None]
    voiced = torch.isfinite(f0_hz) & (f0_hz > 5) & valid
    pitch = torch.where(voiced, f0_hz.clamp_min(5).log2(), 0.)
    if normalization == "utterance":
        count = voiced.sum(1, keepdim=True).clamp_min(1)
        mean = pitch.sum(1, keepdim=True) / count
        variance = ((pitch - mean) * voiced).square().sum(1, keepdim=True) / count
        pitch = torch.where(voiced, (pitch - mean) / variance.clamp_min(1e-8).sqrt(), 0.)
    elif normalization == "global":
        if global_mean is None or global_std is None or global_std <= 0:
            raise ValueError("global normalization requires global_mean and positive global_std in log2 Hz")
        pitch = torch.where(voiced, (pitch - global_mean) / global_std, 0.)
    return pitch, voiced, valid


class PitchCodeHead(nn.Module):
    """Centered temporal MLP predicting one FACodec prosody code per frame."""

    def __init__(self, codebook_size=1024, hidden_sizes=(128,), context_size=5, normalization='none', global_mean=None, global_std=None):
        super().__init__()
        if type(context_size) is not int or context_size < 1 or context_size % 2 == 0:
            raise ValueError("context_size must be a positive odd integer")
        self.context_size = context_size
        self.normalization = normalization
        self.global_mean = global_mean
        self.global_std = global_std
        dims = [2 * context_size, *hidden_sizes, codebook_size]
        layers = []
        for a, b in itertools.pairwise(dims[:-1]):
            layers.extend([nn.Linear(a, b), nn.GELU()])
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.network = nn.Sequential(*layers)

    def forward(self, f0_hz, lengths=None):
        if f0_hz.ndim != 2:
            raise ValueError("Expected F0 [B,T] in Hz")
        log_pitch, voiced, _ = pitch_inputs(f0_hz, lengths, self.normalization, self.global_mean, self.global_std)
        x = torch.stack([log_pitch, voiced.to(log_pitch.dtype)], dim=-1)
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
        self.normalization = normalization
        self.global_mean = global_mean
        self.global_std = global_std
        self.input_projection = nn.Conv1d(2, hidden_size, 1)
        self.blocks = nn.ModuleList([ResidualDilatedBlock(hidden_size, d) for d in dilations])
        self.output_projection = nn.Sequential(
            nn.Conv1d(3 * hidden_size, hidden_size, 1),
            nn.GELU(),
            nn.Conv1d(hidden_size, codebook_size, 1),
        )

    def forward(self, f0_hz, lengths=None):
        if f0_hz.ndim != 2:
            raise ValueError("Expected F0 [B,T] in Hz")
        pitch, voiced, mask = pitch_inputs(f0_hz, lengths, self.normalization, self.global_mean, self.global_std)
        b, t = f0_hz.shape
        if lengths is None:
            lengths = torch.full((b,), t, device=f0_hz.device, dtype=torch.long)
        x = self.input_projection(torch.stack([pitch, voiced.to(pitch.dtype)], dim=1))
        m = mask.unsqueeze(1)
        for block in self.blocks:
            x = block(x * m) * m
        count = lengths.to(x.dtype).view(b, 1, 1)
        mean = x.sum(dim=-1, keepdim=True) / count
        variance = ((x - mean) * m).square().sum(dim=-1, keepdim=True) / count
        std = (variance + 1e-6).sqrt()
        context = torch.cat([x, mean.expand_as(x), std.expand_as(x)], dim=1)
        return self.output_projection(context).transpose(1, 2)
