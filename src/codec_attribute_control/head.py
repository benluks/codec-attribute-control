"""Framewise normalized-F0 probe head."""
import torch
from torch import nn


class LinearF0Head(nn.Module):
    def __init__(self, input_dim=256):
        super().__init__()
        self.projection = nn.Linear(input_dim, 1)

    def forward(self, features):
        if features.ndim != 3:
            raise ValueError("Expected [B,T,C] features")
        return self.projection(features).squeeze(-1)

    def loss(self, predictions, targets, valid):
        if predictions.shape != targets.shape or valid.shape != targets.shape:
            raise ValueError("Predictions, targets and mask must have matching [B,T] shapes")
        if not torch.any(valid):
            return predictions.sum() * 0.0
        return (predictions[valid] - targets[valid]).square().mean()
