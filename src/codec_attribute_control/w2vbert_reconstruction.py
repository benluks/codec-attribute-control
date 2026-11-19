"""Reuse Quick Convert's W2V-BERT and the learned reconstruction layer fusion."""

from pathlib import Path

import torch
from torch import nn
from quick_convert.components.encoders.content.w2vbert import W2VBertContentEncoder
from quick_convert.components.layers.layer_fusion import LayerWeightedSum


class ReconstructionW2VBertEncoder(nn.Module):
    """Frozen 50 Hz, 1024-dimensional representation from an SSLR checkpoint.

    The checkpoint must contain the learned system.feature_transform.weights.
    The original W2V-BERT model is loaded through Quick Convert, not reimplemented.
    """

    def __init__(self, checkpoint_path: str, revision: str = "da985ba0987f70aaeb84a80f2851cfac8c697a7b", device: str = "cuda"):
        super().__init__()
        self.content = W2VBertContentEncoder(layer=None, revision=revision, device=device)
        self.feature_transform = LayerWeightedSum(num_layers=24)
        checkpoint = torch.load(Path(checkpoint_path).expanduser(), map_location="cpu", weights_only=False)
        state = checkpoint.get("state_dict", checkpoint)
        key = "system.feature_transform.weights"
        if key not in state:
            raise KeyError(f"Reconstruction checkpoint is missing {key!r}; available fusion keys: "
                           f"{[k for k in state if 'feature_transform' in k]}")
        self.feature_transform.load_state_dict({"weights": state[key]}, strict=True)
        self.requires_grad_(False)
        self.eval()

    def forward(self, batch):
        features = self.content(batch)
        if features.values.ndim != 4 or features.values.shape[2:] != (24, 1024):
            raise ValueError(f"Expected W2V-BERT [B,T,24,1024], got {tuple(features.values.shape)}")
        features.values = self.feature_transform(features.values)
        features.feature_dim = 1024
        return features
