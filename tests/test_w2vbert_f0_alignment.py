import torch

from codec_attribute_control.training import align_f0_to_feature_grid


def test_align_80hz_pitch_to_50hz_features():
    hz = torch.arange(16, dtype=torch.float).reshape(1, 16)
    values, lengths = align_f0_to_feature_grid(
        hz, torch.tensor([16]), torch.tensor([10]), 10,
        source_hz=80, target_hz=50,
    )
    assert lengths.tolist() == [10]
    assert values.shape == (1, 10)
    assert values[0].tolist() == [0, 2, 3, 5, 6, 8, 10, 11, 13, 14]


def test_alignment_respects_variable_lengths():
    hz = torch.arange(32, dtype=torch.float).reshape(2, 16)
    values, lengths = align_f0_to_feature_grid(
        hz, torch.tensor([16, 8]), torch.tensor([10, 5]), 10,
        source_hz=80, target_hz=50,
    )
    assert lengths.tolist() == [10, 5]
    assert torch.isnan(values[1, 5:]).all()
