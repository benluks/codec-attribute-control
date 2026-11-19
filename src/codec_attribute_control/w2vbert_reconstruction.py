"""Reuse Quick Convert's W2V-BERT and the learned reconstruction layer fusion."""

from dataclasses import replace
from pathlib import Path

import torch
from quick_convert.components.encoders.content.w2vbert import W2VBertContentEncoder
from quick_convert.components.layers.layer_fusion import LayerWeightedSum
from torch import nn


class ReconstructionW2VBertEncoder(nn.Module):
    """Frozen 50 Hz, 1024-dimensional representation from an SSLR checkpoint.

    The checkpoint must contain the learned system.feature_transform.weights.
    The original W2V-BERT model is loaded through Quick Convert, not reimplemented.
    """

    # The first stacked frame spans two 25 ms filterbanks starting 10 ms apart.
    frame_offset_seconds = 0.0175

    def __init__(
        self,
        checkpoint_path: str,
        revision: str = "da985ba0987f70aaeb84a80f2851cfac8c697a7b",
        device: str = "cpu",
    ):
        super().__init__()
        checkpoint = torch.load(
            Path(checkpoint_path).expanduser(),
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
        state = checkpoint.get("state_dict", checkpoint)
        key = "system.feature_transform.weights"
        if key not in state:
            raise KeyError(f"Reconstruction checkpoint is missing {key!r}")

        self.content = W2VBertContentEncoder(
            layer=None, revision=revision, device="cpu"
        )
        prefix = "system.online_encoders.content."
        encoder_state = {
            name.removeprefix(prefix): value
            for name, value in state.items()
            if name.startswith(prefix)
        }
        if encoder_state:
            self.content.load_state_dict(encoder_state, strict=True, assign=True)
        self.feature_transform = LayerWeightedSum(num_layers=24)
        self.feature_transform.load_state_dict({"weights": state[key]}, strict=True)
        self.to(device).requires_grad_(False).eval()

    @property
    def frame_hz(self):
        return self.content.frame_hz

    def forward(self, batch):
        # Quick Convert uses this field for processor output placement.
        self.content.device = next(self.content.parameters()).device
        features = self.content(batch)
        if features.values.ndim != 4 or features.values.shape[2:] != (24, 1024):
            raise ValueError(
                f"Expected W2V-BERT [B,T,24,1024], got {tuple(features.values.shape)}"
            )
        values = self.feature_transform(features.values)
        pad = (
            torch.arange(values.shape[1], device=values.device)
            >= features.lengths[:, None]
        )
        return replace(features, values=values.masked_fill(pad.unsqueeze(-1), 0))
