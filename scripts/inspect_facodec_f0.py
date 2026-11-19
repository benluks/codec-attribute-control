"""Inspect FACodec's own frozen F0 and voiced/unvoiced prediction head.

Run: uv run --extra train --extra facodec python scripts/inspect_facodec_f0.py audio.wav --f0-cache path/to/utt.pt
"""
import argparse

import torch
import torch.nn.functional as F

from codec_attribute_control.facodec import FACodec
from codec_attribute_control.facodec_encoder import FACodecContentEncoder


@torch.inference_mode()
def predict(codec, waveform, device):
    features = FACodecContentEncoder(device=device, codec=codec).encode_waveforms(
        waveform[None].to(device), sample_rate=16000
    )
    prosody = features.streams["prosody"].transpose(1, 2)
    f0, uv = codec.decoder.f0_predictor(prosody)
    return f0.squeeze().detach().cpu().flatten(), uv.squeeze().detach().cpu().flatten()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("audio", help="Mono or multichannel WAV; resampled to 16 kHz")
    p.add_argument("--f0-cache", help="Optional cached JDC F0 tensor in Hz")
    p.add_argument("--output", default="facodec_native_f0.png")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    import torchaudio
    waveform, sr = torchaudio.load(args.audio)
    waveform = waveform.mean(0)
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    codec = FACodec(device=args.device)
    predicted, uv = predict(codec, waveform, args.device)
    print(f"Native predictor F0: {tuple(predicted.shape)}, min={predicted.min():.4f}, max={predicted.max():.4f}, mean={predicted.mean():.4f}")
    print(f"Native predictor UV: {tuple(uv.shape)}, min={uv.min():.4f}, max={uv.max():.4f}")
    print("UV values are raw model outputs; threshold/semantics must be verified against upstream training.")
    print("Native F0 is in the model's training target space, not necessarily Hz.")

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2 if args.f0_cache else 1, 1, figsize=(12, 5), sharex=True)
    axes = ax if args.f0_cache else [ax]
    axes[0].plot(torch.arange(len(predicted)) / 80, predicted.numpy(), label="FACodec native F0 output")
    axes[0].set_ylabel("Native F0 output")
    axes[0].legend()
    if args.f0_cache:
        reference = torch.load(args.f0_cache, map_location="cpu", weights_only=True).float().flatten()
        if len(reference) != len(predicted):
            print(f"WARNING: JDC frames={len(reference)} vs FACodec frames={len(predicted)}; interpolating for visualization only")
            reference = F.interpolate(reference[None, None], size=len(predicted), mode="linear", align_corners=False).flatten()
        voiced = reference > 0
        if voiced.sum() > 1:
            voiced_f0 = reference[voiced]
            normalized = (voiced_f0 - voiced_f0.mean()) / voiced_f0.std().clamp_min(1e-8)
            predicted_voiced = predicted[voiced]
            if predicted_voiced.std() > 0:
                correlation = torch.corrcoef(torch.stack([predicted_voiced, normalized]))[0, 1].item()
                print(f"Correlation with utterance-normalized voiced JDC F0 (diagnostic, not official FACodec metric): {correlation:.4f}")
            axes[1].plot(torch.arange(len(reference))[voiced] / 80, normalized.numpy(), ".", markersize=2, label="JDC voiced F0, utterance z-score")
            axes[1].plot(torch.arange(len(predicted))[voiced] / 80, predicted_voiced.numpy(), ".", markersize=2, label="FACodec native F0 on voiced JDC frames")
            axes[1].legend()
            axes[1].set_ylabel("Comparison (not calibrated)")
        else:
            print("No voiced frames in cached reference")
    axes[-1].set_xlabel("Time (s)")
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
