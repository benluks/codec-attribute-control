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
from .targets import f0_target


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


def _probe_target_mode(checkpoint):
    config = checkpoint_run_config(Path(checkpoint).expanduser())
    mode = OmegaConf.select(OmegaConf.load(config), "target_mode") if config else None
    if OmegaConf.is_dict(mode):
        mode = mode.get("value")
    if mode is None:
        return "log2"
    if mode not in ("log2", "normalized_log2"):
        raise ValueError(f"Unsupported pitch probe target mode: {mode}")
    return mode


def _load_pitch_head(checkpoint, feature_dim, device):
    path = Path(checkpoint).expanduser()
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


def _edit_pitch(
    system,
    batch,
    checkpoint,
    semitones,
    steps,
    target_f0=None,
    target_frame_period=None,
    *,
    target_mode="log2",
):
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
        if target_f0 is None:
            target = shift_log2_f0(head(features), semitones)
        else:
            if target_frame_period is None or target_frame_period <= 0:
                raise ValueError("target_frame_period must be positive")
            contour = torch.as_tensor(
                target_f0, dtype=features.dtype, device=features.device
            )
            if contour.ndim != 1 or contour.numel() < 2:
                raise ValueError("Expected a one-dimensional F0 contour")
            # Browser samples F0 at 80 Hz; W2v-BERT latents are at 50 Hz.
            # Interpolate only inside voiced regions; never bridge unvoiced gaps.
            times = (
                torch.arange(features.shape[1], device=features.device)
                / system.online_encoders["content"].frame_hz
                + 0.0175
            )
            positions = times / target_frame_period
            left = positions.floor().long().clamp(0, contour.numel() - 1)
            right = (left + 1).clamp(max=contour.numel() - 1)
            weight = (positions - left).clamp(0, 1)
            a, b = contour[left], contour[right]
            voiced = torch.isfinite(a) & torch.isfinite(b) & (a > 0) & (b > 0)
            hz = torch.lerp(torch.nan_to_num(a), torch.nan_to_num(b), weight)
            hz = hz[None].masked_fill(~voiced[None], float("nan"))
            if target_mode == "normalized_log2":
                common_voiced = (
                    voiced[None] & torch.isfinite(aligned) & (aligned > 5) & (hz > 5)
                )
                hz = hz.masked_fill(~common_voiced, float("nan"))
                normalized, _ = f0_target(hz[0].float().cpu().numpy())
                target = torch.as_tensor(normalized, device=features.device)[None]
                target = target.masked_fill(~common_voiced, float("nan"))
            else:
                target = torch.log2(hz.clamp_min(1))
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
    target_f0=None,
    target_frame_period: float | None = None,
) -> Path:
    """Restore SSLR and synthesize the source, optionally editing its pitch."""
    if probe_checkpoint is None and (semitones is not None or target_f0 is not None):
        raise ValueError("Pitch editing requires a probe checkpoint")
    if probe_checkpoint is not None and (semitones is None) == (target_f0 is None):
        raise ValueError("Provide exactly one of semitones or target_f0")
    if semitones is not None and not math.isfinite(semitones):
        raise ValueError("The semitone shift must be finite")
    target_mode = _probe_target_mode(probe_checkpoint) if probe_checkpoint else "log2"
    if semitones is not None and target_mode == "normalized_log2":
        raise ValueError(
            "Uniform semitone shifts require a log2 probe; normalized_log2 probes support contour-shape edits"
        )
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
        generation = _edit_pitch(
            system,
            batch,
            probe_checkpoint,
            semitones,
            steps,
            target_f0,
            target_frame_period,
            target_mode=target_mode,
        )
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
