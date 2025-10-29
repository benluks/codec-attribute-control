import argparse
import csv
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import librosa
import numpy as np
import pyworld as pw
import soundfile as sf
from tqdm import tqdm


def list_audio_files(root: Path):
    exts = {".wav", ".flac", ".mp3", ".ogg", ".m4a"}
    return [p for p in root.rglob("*") if p.suffix.lower() in exts and p.is_file()]


def get_output_paths(audio_path: Path, in_root: Path, out_audio_root: Path, out_pitch_root: Path):
    rel = audio_path.relative_to(in_root)
    out_audio_path = (out_audio_root / rel).with_name(f"{rel.stem}_flat{rel.suffix}")
    out_pitch_path = (out_pitch_root / rel).with_suffix(".npy")
    out_audio_path.parent.mkdir(parents=True, exist_ok=True)
    out_pitch_path.parent.mkdir(parents=True, exist_ok=True)
    return out_audio_path, out_pitch_path


def flatten_audio(audio_path: Path, strength: float):
    # Load audio
    audio, sr = librosa.load(audio_path, sr=24000, dtype=np.float64, mono=True)

    # Analyze with pyworld
    _f0, t = pw.harvest(audio, fs=24000, frame_period=32/3)
    f0 = pw.stonemask(audio, _f0, t, fs=24000)
    sp = pw.cheaptrick(audio, f0, t, fs=24000)
    ap = pw.d4c(audio, f0, t, fs=24000)

    # Get voiced frames
    voiced = f0 > 0

    # Compute mean
    mean_f0 = np.mean(f0[voiced]) if np.any(voiced) else 0.0

    # Sample shift and compute flat f0
    semitone_shift = np.random.uniform(low=-abs(strength), high=abs(strength))
    ratio = 2 ** (semitone_shift / 12)
    f0_flat = np.where(voiced, mean_f0 * ratio, 0.0)

    # Resynthesize and trim audio
    warped = pw.synthesize(f0_flat, sp, ap, fs=24000, frame_period=32/3)
    warped = warped[: len(audio)]

    # Get log-f0
    log_f0 = np.where(voiced, np.log2(f0), 0.0)

    return warped, log_f0


def process_file(audio_path: Path, in_root: Path, out_audio_root: Path, out_pitch_root: Path, strength: float, csv_writer, lock):
    out_audio_path, out_pitch_path = get_output_paths(audio_path, in_root, out_audio_root, out_pitch_root)
    warped, log_f0 = flatten_audio(audio_path, strength)
    sf.write(out_audio_path, warped, samplerate=24000)
    np.save(out_pitch_path, log_f0)

    audio_rel = str(audio_path)
    pitch_rel = str(out_pitch_path)
    flat_rel = str(out_audio_path)

    # Thread-safe CSV writing
    with lock:
        csv_writer.writerow([audio_rel, pitch_rel, flat_rel])


def main():
    parser = argparse.ArgumentParser(prog="Preprocess dataset", description="Perturb audios with stochastic pitch flattening. Saves the new audios and original log-f0")
    parser.add_argument("input_dir", type=str, help="Path to directory containing audios")
    parser.add_argument("output_dir", type=str, help="Target directory where 'perturbed' and 'pitch' will be created")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file to write results")
    parser.add_argument("--strength", type=float, default=5.0, help="Perturbation strength in semitones (uniform in [-s, +s])")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Number of worker threads")
    args = parser.parse_args()

    in_root = Path(args.input_dir).resolve()
    out_root = Path(args.output_dir).resolve()
    out_audio_root = out_root / "perturbed"
    out_pitch_root = out_root / "pitch"
    out_audio_root.mkdir(parents=True, exist_ok=True)
    out_pitch_root.mkdir(parents=True, exist_ok=True)

    files = list_audio_files(in_root)
    csv_path = Path(args.csv).resolve()
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    lock = threading.Lock()
    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["audio_path", "pitch_path", "flat_path"])

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_file, p, in_root, out_audio_root, out_pitch_root, args.strength, writer, lock): p for p in files}
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing"):
                try:
                    f.result()
                except Exception as e:
                    print(f"Error processing {futures[f]}: {e}")


if __name__ == "__main__":
    main()
