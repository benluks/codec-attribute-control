import torch
from codec_attribute_control.supervision import prepare_f0_targets


def test_variable_length_and_unvoiced_masks():
    hz = torch.tensor([[100., 110., 120., 0., 999.],
                       [200., 220., 240., 260., 280.]])
    targets, mask = prepare_f0_targets(hz, torch.tensor([4, 5]))
    assert targets.shape == hz.shape
    assert mask.tolist() == [[True, True, True, False, False],
                             [True, True, True, True, True]]
    assert targets[0, 4] == 0


def test_normalization_is_per_segment():
    hz = torch.tensor([[100., 110., 120., 130., 140.]])
    a, _ = prepare_f0_targets(hz, torch.tensor([3]))
    b, _ = prepare_f0_targets(hz, torch.tensor([5]))
    assert not torch.allclose(a[0, :3], b[0, :3])


def test_constant_pitch_has_no_valid_targets():
    _, mask = prepare_f0_targets(torch.full((1, 4), 100.), torch.tensor([4]))
    assert not mask.any()
