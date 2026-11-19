import pytest
import torch

from codec_attribute_control.head import MLPF0Head, FACodecCNNF0Head


@pytest.mark.parametrize("hidden_sizes", [[], [128]])
def test_framewise_heads_share_masked_loss(hidden_sizes):
    head = MLPF0Head(hidden_sizes=hidden_sizes)
    features = torch.randn(2, 9, 256, requires_grad=True)
    predicted = head(features)
    mask = torch.tensor([[True] * 5 + [False] * 4, [True] * 9])
    loss = head.loss(predicted, torch.zeros_like(predicted), mask)
    loss.backward()
    assert features.grad is not None


def test_facodec_cnn_matches_original_predictor():
    pytest.importorskip("Amphion")
    head = FACodecCNNF0Head()
    features = torch.randn(2, 40, 256, requires_grad=True)
    predictions = head(features)
    assert predictions.shape == (2, 40)
    valid = torch.ones_like(predictions, dtype=torch.bool)
    head.loss(predictions, torch.zeros_like(predictions), valid).backward()
    assert features.grad is not None
    assert len(head.predictor.heads) == 2


def test_all_heads_reject_bad_input_shape():
    with pytest.raises(ValueError):
        MLPF0Head()(torch.randn(2, 256))


@pytest.mark.parametrize("context_size", [1, 3, 5])
@pytest.mark.parametrize("hidden_sizes", [[], [128]])
def test_contextual_mlp_shape_and_gradients(context_size, hidden_sizes):
    head = MLPF0Head(input_dim=2, hidden_sizes=hidden_sizes, context_size=context_size)
    features = torch.randn(2, 8, 2, requires_grad=True)
    assert head(features).shape == (2, 8)
    head(features).sum().backward()
    assert features.grad is not None


def test_context_window_center_alignment():
    head = MLPF0Head(input_dim=1, context_size=3)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[1., 10., 100.]]))
        head.projection.bias.zero_()
    actual = head(torch.tensor([[[1.], [2.], [3.], [4.]]]))
    assert torch.allclose(actual, torch.tensor([[211., 321., 432., 443.]]))


@pytest.mark.parametrize("context_size", [0, 2, -1, True])
def test_invalid_context_size(context_size):
    with pytest.raises(ValueError, match="context_size"):
        MLPF0Head(context_size=context_size)
