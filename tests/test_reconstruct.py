from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from codec_attribute_control import reconstruct


def test_migrates_wandb_wrapped_sslr_config(tmp_path, monkeypatch):
    original = tmp_path / "config.yaml"
    OmegaConf.save(
        {
            "system": {
                "value": {
                    "_target_": "quick_convert.systems.reconstruction.SSLReconstructionSystem",
                    "decoder": {
                        "_target_": "quick_convert.components.decoders.CosyVoiceSpectrogramGenerator"
                    },
                    "online_encoders": {
                        "content": {
                            "_target_": "quick_convert.components.ssl.W2VBertContentEncoder",
                            "revision": "pinned",
                        }
                    },
                }
            },
        },
        original,
    )
    before = original.read_bytes()

    def load(source, *, config, map_location):
        assert source == tmp_path
        assert map_location == "cpu"
        system = OmegaConf.load(config).system
        assert (
            system.decoder._target_
            == "quick_convert.components.generators.acoustic.CosyVoiceSpectrogramGenerator"
        )
        assert (
            system.online_encoders.content._target_
            == "quick_convert.components.encoders.content.w2vbert.W2VBertContentEncoder"
        )
        assert system.online_encoders.content.revision == "pinned"
        return config

    monkeypatch.setattr(reconstruct, "load_inference_system", load)
    temporary = reconstruct._load_sslr(tmp_path, None)
    assert not Path(temporary).exists()
    assert original.read_bytes() == before


def test_pitch_edit_reaches_decoder_with_same_lengths_and_speaker(monkeypatch):
    source = torch.zeros(1, 2, 1)
    lengths = torch.tensor([2])
    speaker = torch.ones(1, 2)
    result = object()

    class System:
        def __init__(self):
            self.online_encoders = {"content": SimpleNamespace(frame_hz=50)}

        def get_resource(self, batch, name):
            return SimpleNamespace(values=source, lengths=lengths, metadata=None)

        def encode_features(self, features, lengths, **_):
            return features, lengths, None

        def speaker_condition(self, batch):
            return speaker

        def decode_features(self, features, output_lengths, embedding, *, run_vocoder):
            assert run_vocoder
            assert output_lengths is lengths
            assert embedding is speaker
            assert features[0, 0, 0] > 0
            assert features[0, 1, 0] == 0
            assert torch.count_nonzero(source) == 0
            return result

    monkeypatch.setattr(
        reconstruct,
        "WORLDF0Extractor",
        lambda: SimpleNamespace(
            extract=lambda *args: torch.tensor([0.0, 100.0, 0.0, 0.0])
        ),
    )
    monkeypatch.setattr(
        reconstruct,
        "_load_pitch_head",
        lambda *args: torch.nn.Flatten(1),
    )
    batch = SimpleNamespace(
        waveforms=torch.zeros(1, 640),
        lengths=torch.tensor([640]),
        sample_rates=torch.tensor([16000]),
    )
    assert reconstruct._edit_pitch(System(), batch, "probe.ckpt", 3, 4) is result


def test_normalized_probe_rejects_semitone_shift_before_loading_system(tmp_path):
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    probe = checkpoints / "best.ckpt"
    probe.touch()
    OmegaConf.save(
        {"target_mode": {"value": "normalized_log2"}}, tmp_path / "config.yaml"
    )
    with pytest.raises(
        ValueError, match="Uniform semitone shifts require a log2 probe"
    ):
        reconstruct.reconstruct_identity(
            "source.wav", "model.ckpt", "out.wav", probe_checkpoint=probe, semitones=3
        )
