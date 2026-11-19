from types import SimpleNamespace

import torch
from quick_convert.components.encoders.content.base import ContentFeatures

from codec_attribute_control.w2vbert_reconstruction import ReconstructionW2VBertEncoder


def test_reconstruction_encoder_restores_backbone_and_fusion(tmp_path, monkeypatch):
    class Backbone(torch.nn.Module):
        frame_hz = 50

        def __init__(self, **kwargs):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(self, batch):
            values = torch.arange(24.0).view(1, 1, 24, 1).expand(1, 2, 24, 1024)
            return ContentFeatures(
                values=values * self.scale,
                lengths=torch.tensor([1]),
                feature_dim=1024,
                representation_type="continuous",
                temporal_granularity="frame",
                backend="test",
                model_name="test",
                layer=None,
                frame_hz=50,
            )

    monkeypatch.setattr(
        "codec_attribute_control.w2vbert_reconstruction.W2VBertContentEncoder", Backbone
    )
    path = tmp_path / "reconstruction.ckpt"
    weights = torch.arange(24.0).reshape(1, 24)
    torch.save(
        {
            "state_dict": {
                "system.online_encoders.content.scale": torch.tensor(2.0),
                "system.feature_transform.weights": weights,
            }
        },
        path,
    )
    encoder = ReconstructionW2VBertEncoder(str(path))
    assert not encoder.training
    assert all(not p.requires_grad for p in encoder.parameters())
    actual = encoder(SimpleNamespace())
    expected = (2 * torch.arange(24.0) * weights[0].softmax(0)).sum()
    torch.testing.assert_close(actual.values[0, 0], expected.expand(1024))
    assert torch.count_nonzero(actual.values[0, 1]) == 0
    assert actual.lengths.tolist() == [1]
    assert actual.frame_hz == 50
