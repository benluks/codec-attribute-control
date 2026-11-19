import pytest
import torch

from codec_attribute_control.head import LinearF0Head


@pytest.mark.parametrize("hidden_sizes", [[], [128], [128, 64]])
@pytest.mark.parametrize("activation", ["gelu", "relu", "tanh", "silu"])
def test_probe_shapes_and_gradients(hidden_sizes, activation):
    head = LinearF0Head(input_dim=256, hidden_sizes=hidden_sizes, activation=activation)
    x = torch.randn(2, 7, 256, requires_grad=True)
    y = head(x)
    assert y.shape == (2, 7)
    y.square().mean().backward()
    assert x.grad is not None
    assert len([m for m in head.modules() if isinstance(m, torch.nn.Linear)]) == len(hidden_sizes) + 1


def test_default_is_single_linear_layer():
    head = LinearF0Head()
    assert isinstance(head.projection, torch.nn.Linear)


@pytest.mark.parametrize("hidden_sizes", [[0], [-1], [2.5], [True]])
def test_invalid_hidden_sizes(hidden_sizes):
    with pytest.raises(ValueError):
        LinearF0Head(hidden_sizes=hidden_sizes)


def test_invalid_activation():
    with pytest.raises(ValueError, match="Unsupported activation"):
        LinearF0Head(activation="unknown")
