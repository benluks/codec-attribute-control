"""Interchangeable F0 regression heads with shared masked loss."""
import torch
from torch import nn


class MaskedF0RegressionHead(nn.Module):
    """Common voiced-frame MSE objective for all F0 heads."""

    def loss(self, predictions, targets, valid):
        if predictions.shape != targets.shape or valid.shape != targets.shape:
            raise ValueError("Predictions, targets and mask must have matching [B,T] shapes")
        if not torch.any(valid):
            return predictions.sum() * 0.0
        return (predictions[valid] - targets[valid]).square().mean()


class LinearF0Head(MaskedF0RegressionHead):
    """Linear F0 probe, optionally extended with hidden MLP layers."""

    def __init__(self, input_dim=256, hidden_sizes=None, activation="gelu"):
        super().__init__()
        hidden_sizes = [] if hidden_sizes is None else list(hidden_sizes)
        if any(not isinstance(size, int) or isinstance(size, bool) or size <= 0 for size in hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
        activations = {
            "gelu": nn.GELU,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "silu": nn.SiLU,
        }
        if activation.lower() not in activations:
            raise ValueError(f"Unsupported activation {activation!r}; choose from {sorted(activations)}")
        layers = []
        previous = input_dim
        for size in hidden_sizes:
            layers.extend([nn.Linear(previous, size), activations[activation.lower()]()])
            previous = size
        layers.append(nn.Linear(previous, 1))
        self.projection = nn.Sequential(*layers) if hidden_sizes else layers[0]

    def forward(self, features):
        if features.ndim != 3:
            raise ValueError("Expected [B,T,C] features")
        return self.projection(features).squeeze(-1)


class FACodecCNNF0Head(MaskedF0RegressionHead):
    """FACodec's original temporal CNN F0 architecture, initialized from scratch.

    Amphion's `CNNLSTM` is a historical name: the module contains three
    dilated residual convolution units and SnakeBeta, but no recurrent LSTM.
    The original predictor emits F0 and voicing; we use its F0 output only.
    """

    def __init__(self, input_dim=256):
        super().__init__()
        from .amphion_source import ensure_amphion_importable

        ensure_amphion_importable()
        from Amphion.models.codec.ns3_codec.facodec import CNNLSTM

        self.predictor = CNNLSTM(input_dim, 1, 2)

    def forward(self, features):
        if features.ndim != 3:
            raise ValueError("Expected [B,T,C] features")
        f0, _uv = self.predictor(features.transpose(1, 2))
        return f0.squeeze(-1)
