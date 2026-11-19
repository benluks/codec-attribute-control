import pytest
import torch

from codec_attribute_control.head import LinearF0Head, FACodecCNNF0Head


@pytest.mark.parametrize("hidden_sizes", [[], [128]])
def test_framewise_heads_share_masked_loss(hidden_sizes):
    head = LinearF0Head(hidden_sizes=hidden_sizes)
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
        LinearF0Head()(torch.randn(2, 256))
