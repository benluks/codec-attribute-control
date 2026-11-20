"""Identity reconstruction smoke test using Quick Convert's native inference loader.

This deliberately does not implement pitch editing: first verify that a saved
SSLR model can reconstruct a source utterance without latent modification.
"""
import argparse
from dataclasses import replace
from pathlib import Path

import torch
from quick_convert.data import AudioBatch
from quick_convert.inference import load_inference_system
from quick_convert.systems.conversion.ssl_reconstruction import SSLReconstructionConversionSystem
from quick_convert.utils.audio import load_audio_input


def make_audio_batch(path: str | Path, sample_rate: int = 16000) -> AudioBatch:
    path = Path(path).expanduser().resolve()
    waveform = load_audio_input(path, target_sample_rate=sample_rate, mono=True).reshape(-1)
    return AudioBatch(
        utt_ids=[path.stem],
        paths=[path],
        splits=[None],
        resources={},
        waveforms=waveform.unsqueeze(0),
        lengths=torch.tensor([waveform.numel()], dtype=torch.long),
        sample_rates=torch.tensor([sample_rate], dtype=torch.long),
    )


@torch.inference_mode()
def reconstruct_identity(
    source: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    *,
    device: str = "cpu",
    config: str | Path | None = None,
) -> Path:
    """Load saved SSLR weights/config and synthesize without changing features."""
    system = load_inference_system(
        checkpoint, config=config, map_location="cpu"
    )
    if not isinstance(system, SSLReconstructionConversionSystem):
        raise TypeError(f"Expected SSL reconstruction system, got {type(system).__name__}")
    system = system.to(device).eval()
    sample_rate = system.online_encoders["content"].sample_rate
    batch = make_audio_batch(source, sample_rate=sample_rate)
    batch = replace(batch, waveforms=batch.waveforms.to(device), lengths=batch.lengths.to(device), sample_rates=batch.sample_rates.to(device))
    result = system(batch, run_vocoder=True)
    if result.generation.audio is None:
        raise RuntimeError("SSLR decoder did not return waveform; verify vocoder configuration")
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    system.save_generation(result.generation, output)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct audio from a saved SSLR model")
    parser.add_argument("--source", required=True, help="Input audio path")
    parser.add_argument("--checkpoint", required=True, help="SSLR run directory, checkpoint or export")
    parser.add_argument("--output", required=True, help="Output WAV path")
    parser.add_argument("--config", default=None, help="Optional saved system config override")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(reconstruct_identity(args.source, args.checkpoint, args.output, device=args.device, config=args.config))


if __name__ == "__main__":
    main()
