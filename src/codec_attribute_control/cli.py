"""Minimal command line for the FACodec F0 experiment."""
import argparse
import json

import torch

from .experiment import intervene, train


def main():
    parser = argparse.ArgumentParser(description="FACodec post-VQ prosody F0 probe")
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser("train")
    fit.add_argument("--audio-dir", required=True)
    fit.add_argument("--output", default="outputs/probe.joblib")
    fit.add_argument("--alpha", type=float, default=100.0)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--max-recordings", type=int, default=50)
    fit.add_argument("--f0-backend", choices=["pyin", "yaapt"], default="pyin")
    edit = commands.add_parser("intervene")
    edit.add_argument("--audio", required=True)
    edit.add_argument("--probe", default="outputs/probe.joblib")
    edit.add_argument("--output-dir", default="outputs/interventions")
    edit.add_argument("--semitones", type=float, nargs="+", default=[-6, -3, 0, 3, 6])
    for cmd in (fit, edit):
        cmd.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.command == "train":
        result = train(args.audio_dir, args.output, backend=args.f0_backend,
                       alpha=args.alpha, seed=args.seed, max_recordings=args.max_recordings,
                       device=args.device)
    else:
        result = intervene(args.audio, args.probe, args.output_dir,
                           shifts=args.semitones, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
