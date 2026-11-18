import torch

from codec_attribute_control.metrics import F0ProbeMetrics


def test_frame_weighted_errors_and_utterance_correlation():
    metric = F0ProbeMetrics()
    metric.update(
        torch.tensor([[1., 2., 3.], [2., 4., 0.]]),
        torch.tensor([[1., 2., 3.], [1., 2., 0.]]),
        torch.tensor([[True, True, True], [True, True, False]]),
    )
    result = metric.compute()
    assert torch.isclose(result["normalized_log2_f0_mse"], torch.tensor(1.0, dtype=torch.float64))
    assert torch.isclose(result["normalized_log2_f0_mae"], torch.tensor(0.6, dtype=torch.float64))
    assert torch.isclose(result["normalized_log2_f0_pearson"], torch.tensor(1.0, dtype=torch.float64))
    assert result["voiced_frames"] == 5
    assert result["pearson_utterances"] == 2


def test_empty_mask_and_constant_contour():
    metric = F0ProbeMetrics()
    metric.update(
        torch.tensor([[1., 1.], [0., 0.]]),
        torch.tensor([[2., 3.], [0., 0.]]),
        torch.tensor([[True, True], [False, False]]),
    )
    result = metric.compute()
    assert result["voiced_frames"] == 2
    assert result["pearson_utterances"] == 0
    assert torch.isfinite(result["normalized_log2_f0_pearson"])
