import torch

from codec_attribute_control.training import align_f0_to_feature_grid


def test_align_80hz_pitch_to_50hz_features():
    hz = torch.arange(16, dtype=torch.float).reshape(1, 16)
    values, lengths = align_f0_to_feature_grid(
        hz,
        torch.tensor([16]),
        torch.tensor([10]),
        10,
        source_hz=80,
        target_hz=50,
    )
    assert lengths.tolist() == [10]
    assert values.shape == (1, 10)
    assert values[0].tolist() == [0, 2, 3, 5, 6, 8, 10, 11, 13, 14]


def test_alignment_respects_variable_lengths():
    hz = torch.arange(32, dtype=torch.float).reshape(2, 16)
    values, lengths = align_f0_to_feature_grid(
        hz,
        torch.tensor([16, 8]),
        torch.tensor([10, 5]),
        10,
        source_hz=80,
        target_hz=50,
    )
    assert lengths.tolist() == [10, 5]
    assert torch.isnan(values[1, 5:]).all()


def test_training_step_aligns_at_w2vbert_filterbank_centers():
    from types import SimpleNamespace

    from codec_attribute_control.head import LinearF0Head
    from codec_attribute_control.training import F0ProbeTrainingModule

    # Two frames centered at 17.5 and 37.5 ms select F0 indices 1 and 3.
    predictions = torch.tensor([[0.0, 8.0], [0.0, 0.0]], requires_grad=True)
    feature_lengths = torch.tensor([2, 1])
    resource = SimpleNamespace(
        values=torch.tensor(
            [[[100.0], [float("nan")], [100.0], [256.0]], [[1.0], [2.0], [3.0], [4.0]]]
        ),
        lengths=torch.tensor([4, 0]),
    )

    class System(torch.nn.Module):
        encoder = SimpleNamespace(frame_hz=50, frame_offset_seconds=0.0175)
        head = LinearF0Head(input_dim=4)

        def forward(self, batch):
            return predictions, feature_lengths

        def get_resource(self, batch, name):
            return resource

    module = F0ProbeTrainingModule(
        system=System(),
        optimization=None,
        target_mode="log2",
        align_f0_to_features=True,
    )
    module.log = lambda *args, **kwargs: None
    loss = module._shared_step(None, "train").loss
    assert loss.item() == 0.0
    assert module.train_metrics.voiced_frames.item() == 1
    loss.backward()
    assert predictions.grad is not None
