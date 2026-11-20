"""Reconstruct speech and optionally edit pitch with a frozen SSLR probe."""

import argparse
import math
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from omegaconf import OmegaConf
from quick_convert.data import AudioBatch
from quick_convert.inference import load_inference_system
from quick_convert.systems.conversion.ssl_reconstruction import (
    SSLReconstructionConversionSystem,
)
from quick_convert.utils.audio import load_audio_input
from quick_convert.utils.paths import checkpoint_run_config

from .extractors import WORLDF0Extractor
from .f0 import align_f0_to_feature_grid
from .head import FACodecCNNF0Head
from .latent_inversion import invert_pitch, shift_log2_f0


def make_audio_batch(path: str | Path, sample_rate: int = 16000) -> AudioBatch:
    path = Path(path).expanduser().resolve()
    waveform = load_audio_input(
        path, target_sample_rate=sample_rate, mono=True
    ).reshape(-1)
    return AudioBatch(
        utt_ids=[path.stem],
        paths=[path],
        splits=[None],
        resources={},
        waveforms=waveform.unsqueeze(0),
        lengths=torch.tensor([waveform.numel()], dtype=torch.long),
        sample_rates=torch.tensor([sample_rate], dtype=torch.long),
    )


_LEGACY_TARGETS = {
    "quick_convert.components.ssl.W2VBertContentEncoder": "quick_convert.components.encoders.content.w2vbert.W2VBertContentEncoder",
    "quick_convert.components.speaker.speaker_encoders.CosyVoiceCAMPPlusSpeakerEncoder": "quick_convert.components.encoders.speaker.cosyvoice_campplus.CosyVoiceCAMPPlusSpeakerEncoder",
    "quick_convert.components.decoders.CosyVoiceSpectrogramGenerator": "quick_convert.components.generators.acoustic.CosyVoiceSpectrogramGenerator",
}


def _load_sslr(checkpoint, config):
    source = Path(checkpoint).expanduser()
    config_path = (
        Path(config).expanduser()
        if config
        else (
            source / "config.yaml" if source.is_dir() else checkpoint_run_config(source)
        )
    )
    if (
        config_path is not None
        and not config_path.is_absolute()
        and config
        and source.is_dir()
    ):
        config_path = source / config_path
    if config_path is not None and config_path.is_file():
        cfg = OmegaConf.load(config_path)
        system = OmegaConf.select(cfg, "system")
        wrapped = system is not None and "_target_" not in system and "value" in system
        if wrapped:
            system = system.value
        if system is not None:
            migrated = False
            for node in (
                system.get("decoder"),
                *(system.get("online_encoders") or {}).values(),
            ):
                if node is not None and node.get("_target_") in _LEGACY_TARGETS:
                    node._target_ = _LEGACY_TARGETS[node._target_]
                    migrated = True
            if migrated or wrapped:
                composition = OmegaConf.to_container(system, resolve=True)
                with TemporaryDirectory() as directory:
                    override = Path(directory) / "config.yaml"
                    OmegaConf.save({"system": composition}, override)
                    return load_inference_system(
                        source, config=override, map_location="cpu"
                    )
    return load_inference_system(source, config=config, map_location="cpu")


def _load_pitch_head(checkpoint, feature_dim, device):
    path = Path(checkpoint).expanduser()
    config = checkpoint_run_config(path)
    if config is not None:
        mode = OmegaConf.select(OmegaConf.load(config), "target_mode")
        if OmegaConf.is_dict(mode):
            mode = mode.get("value")
        if mode is not None and mode != "log2":
            raise ValueError("Pitch editing requires an absolute log2 F0 CNN probe")
    state = torch.load(path, map_location="cpu", weights_only=False, mmap=True)[
        "state_dict"
    ]
    head = FACodecCNNF0Head(input_dim=feature_dim)
    prefix = "system.head."
    head.load_state_dict(
        {
            key.removeprefix(prefix): value
            for key, value in state.items()
            if key.startswith(prefix)
        },
        strict=True,
    )
    return head.to(device).eval()


def _edit_pitch(system, batch, checkpoint, semitones, steps):
    with torch.no_grad():
        content = system.get_resource(batch, "content")
        features, lengths, _ = system.encode_features(
            content.values, content.lengths, metadata=content.metadata
        )
        speaker = system.speaker_condition(batch)
        head = _load_pitch_head(checkpoint, features.shape[-1], features.device)
        wav = batch.waveforms[0, : int(batch.lengths[0])]
        f0 = (
            WORLDF0Extractor()
            .extract(wav, int(batch.sample_rates[0]))
            .to(features.device)[None]
        )
        aligned, _ = align_f0_to_feature_grid(
            f0,
            lengths.new_tensor([f0.shape[1]]),
            lengths,
            features.shape[1],
            target_hz=system.online_encoders["content"].frame_hz,
            target_offset_seconds=0.0175,
        )
        target = shift_log2_f0(head(features), semitones)
        target = target.masked_fill(
            ~torch.isfinite(aligned) | (aligned <= 5), float("nan")
        )
    if not torch.isfinite(target).any():
        raise ValueError("The source has no voiced frames to edit")
    edited = invert_pitch(head, features.clone(), target, steps=steps)
    with torch.inference_mode():
        return system.decode_features(
            edited.latents, lengths, speaker, run_vocoder=True
        )


def reconstruct_identity(
    source: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    config: str | Path | None = None,
    probe_checkpoint: str | Path | None = None,
    semitones: float | None = None,
    steps: int = 80,
) -> Path:
    """Restore SSLR and synthesize the source, optionally editing its pitch."""
    if (probe_checkpoint is None) != (semitones is None):
        raise ValueError("Provide both a probe checkpoint and a semitone shift")
    if semitones is not None and not math.isfinite(semitones):
        raise ValueError("The semitone shift must be finite")
    system = _load_sslr(checkpoint, config)
    if not isinstance(system, SSLReconstructionConversionSystem):
        raise TypeError(
            f"Expected SSL reconstruction system, got {type(system).__name__}"
        )
    system = system.to(device).eval()
    content = system.online_encoders["content"]
    # W2V-BERT sends processor outputs to its explicitly stored device.
    content.device = torch.device(device)
    batch = make_audio_batch(source, sample_rate=content.sample_rate)
    batch = replace(
        batch,
        waveforms=batch.waveforms.to(device),
        lengths=batch.lengths.to(device),
        sample_rates=batch.sample_rates.to(device),
    )
    if probe_checkpoint is None:
        with torch.inference_mode():
            generation = system(batch, run_vocoder=True).generation
    else:
        generation = _edit_pitch(system, batch, probe_checkpoint, semitones, steps)
    if generation.audio is None:
        raise RuntimeError(
            "SSLR decoder did not return waveform; verify vocoder configuration"
        )
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    system.save_generation(generation, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reconstruct audio from a saved SSLR model"
    )
    parser.add_argument("--source", required=True, help="Input audio path")
    parser.add_argument(
        "--checkpoint", required=True, help="SSLR run directory, checkpoint or export"
    )
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument(
        "--config", default=None, help="Optional saved system config override"
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--probe-checkpoint", help="Trained absolute log2 F0 CNN probe checkpoint"
    )
    parser.add_argument(
        "--semitones", type=float, help="Pitch shift to request from the probe"
    )
    parser.add_argument("--steps", type=int, default=80, help="Adam inversion steps")
    args = parser.parse_args()
    if (args.probe_checkpoint is None) != (args.semitones is None):
        parser.error("--probe-checkpoint and --semitones must be supplied together")
    print(
        reconstruct_identity(
            args.source,
            args.checkpoint,
            args.output,
            device=args.device,
            config=args.config,
            probe_checkpoint=args.probe_checkpoint,
            semitones=args.semitones,
            steps=args.steps,
        )
    )


if __name__ == "__main__":
    main()
