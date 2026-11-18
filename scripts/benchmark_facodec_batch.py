"""Opt-in end-to-end adapter throughput comparison with pretrained FACodec."""
import argparse
import statistics
import time

import torch

from codec_attribute_control.facodec import FACodec
from codec_attribute_control.facodec_encoder import FACodecContentEncoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--mode", choices=["equal", "mixed"], default="equal")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    codec = FACodec(device=device)
    encoder = FACodecContentEncoder(codec=codec)
    base = int(args.seconds * 16000)
    lengths = torch.tensor([
        base if args.mode == "equal" else max(200, base - (i % 4) * 2000)
        for i in range(args.batch_size)
    ])
    waveforms = torch.randn(args.batch_size, base) * 0.08

    def synchronize():
        if device == "cuda":
            torch.cuda.synchronize()

    def old():
        return [codec.encode(waveforms[i, :int(n)]) for i, n in enumerate(lengths)]

    def new():
        return encoder.encode_waveforms(waveforms, lengths, 16000)

    for label, fn in (("per_utterance", old), ("full_batch", new)):
        fn()
        synchronize()
        times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            fn()
            synchronize()
            times.append(time.perf_counter() - start)
        print(f"{label}: median={statistics.median(times):.4f}s, "
              f"audio_seconds_per_second={lengths.sum().item()/16000/statistics.median(times):.2f}")


if __name__ == "__main__":
    main()
