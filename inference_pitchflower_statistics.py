import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import pyworld as pw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pitchflower.synthesizer import PitchFlowerSynthesizer
from vocos import Vocos

SOURCE_DIR = Path('/medias/speech/data/LibriSpeech/LibriSpeech/test-clean')
OUTPUT_DIR = Path('/medias/speech/projects/garga/hackathon/pitchflower_statistics')
NUM_UTTERANCES = 100
SAMPLE_RATE = 24000
STEPS = 10
SEED = 42
SAVE_PLOTS = True

# name, mean shift (in source log-F0 std units), std multiplier
PITCH_CONFIGS = [
    ('baseline',        0.0, 1.0),
    ('mean_down_1std',  -1.0, 1.0),
    ('mean_up_1std',     1.0, 1.0),
    ('mean_down_2std',  -2.0, 1.0),
    ('mean_up_2std',     2.0, 1.0),
    ('std_half',        0.0, 0.5),
    ('std_double',      0.0, 2.0),
    ('std_flat',        0.0, 0.0),
]


def extract_f0(wav):
    audio = wav.reshape(-1).cpu().numpy().astype(np.float64)
    f0, time = pw.harvest(audio, fs=SAMPLE_RATE,
                          frame_period=32 / 3, f0_floor=20)
    return pw.stonemask(audio, f0, time, SAMPLE_RATE), time


def modify_f0(f0, alpha, beta):
    target = f0.copy()
    voiced = np.isfinite(f0) & (f0 > 0)
    if voiced.any():
        y = np.log2(f0[voiced])
        mu, sigma = y.mean(), y.std()
        target[voiced] = 2 ** (mu + alpha * sigma + beta * (y - mu))
    target[~voiced] = 0
    return target


def f0_stats(f0, prefix):
    f0 = np.asarray(f0)
    voiced = np.isfinite(f0) & (f0 > 0)
    values = f0[voiced]
    result = {
        'n_frames': len(f0),
        'n_voiced': int(voiced.sum()),
        'voiced_fraction': float(voiced.mean()) if len(f0) else np.nan,
    }
    names = ['mean_hz', 'std_hz', 'median_hz', 'min_hz', 'max_hz',
             'mean_log2', 'std_log2', 'geometric_mean_hz', 'std_semitones']
    if len(values):
        y = np.log2(values)
        result.update(zip(names, map(float, [values.mean(), values.std(),
                      np.median(values), values.min(), values.max(),
                      y.mean(), y.std(), 2 ** y.mean(), 12 * y.std()])))
    else:
        result.update({name: np.nan for name in names})
    return {f'{prefix}_{k}': v for k, v in result.items()}


def contour_error(target, generated):
    # Identical WORLD frame periods; compare shared prefix on jointly voiced frames.
    n = min(len(target), len(generated))
    a, b = target[:n], generated[:n]
    av = np.isfinite(a) & (a > 0)
    bv = np.isfinite(b) & (b > 0)
    joint = av & bv
    mae = (12 * np.mean(np.abs(np.log2(b[joint]) - np.log2(a[joint])))
           if joint.any() else np.nan)
    return {'f0_mae_semitones': float(mae), 'joint_voiced_frames': int(joint.sum()),
            'voicing_mismatch_fraction': float(np.mean(av != bv)) if n else np.nan}


def write_summary(path, rows):
    if not rows:
        return
    tmp = path.with_suffix('.tmp')
    with tmp.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(next(iter(rows.values())).keys()))
        writer.writeheader()
        writer.writerows(rows.values())
    tmp.replace(path)


def plot_f0(path, utt_id, name, source, time, target, generated, gen_time):
    fig, ax = plt.subplots(figsize=(12, 4))
    # Do not connect voiced regions across unvoiced frames.
    for t, f0, label, ls in [(time, source, 'Source', '-'),
                             (time, target, 'Target', '-'),
                             (gen_time, generated, 'Generated', '--')]:
        ax.plot(t, np.where(f0 > 0, f0, np.nan), label=label, linestyle=ls)
    ax.set(xlabel='Time (s)', ylabel='F0 (Hz)', title=f'{utt_id}: {name}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=NUM_UTTERANCES)
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--manifest', type=Path, default=None,
                        help='Optionally reuse the exact previous selected_utterances.txt')
    args = parser.parse_args()
    if not 1 <= args.limit <= NUM_UTTERANCES:
        parser.error(f'--limit must be between 1 and {NUM_UTTERANCES}')
    save_plots = SAVE_PLOTS and not args.no_plots

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT_DIR / 'selected_utterances.txt'
    if args.manifest is not None:
        paths = [Path(p) for p in args.manifest.read_text().splitlines() if p.strip()]
    elif manifest.exists():
        paths = [Path(p) for p in manifest.read_text().splitlines() if p.strip()]
    else:
        paths = sorted(SOURCE_DIR.rglob('*.flac'))
        random.Random(SEED).shuffle(paths)
    paths = paths[:NUM_UTTERANCES]
    if len(paths) < NUM_UTTERANCES or not all(p.is_file() for p in paths):
        raise RuntimeError('Expected 100 existing source FLAC paths. Check source directory/manifest.')
    if manifest.exists() and manifest.read_text().splitlines() != [str(p) for p in paths]:
        raise RuntimeError('Existing output manifest differs; use a new output directory.')
    manifest.write_text(''.join(f'{p}\n' for p in paths))
    paths = paths[:args.limit]

    (OUTPUT_DIR / 'original').mkdir(exist_ok=True)
    for name, _, _ in PITCH_CONFIGS:
        (OUTPUT_DIR / name).mkdir(exist_ok=True)
    (OUTPUT_DIR / 'contours').mkdir(exist_ok=True)
    if save_plots:
        (OUTPUT_DIR / 'plots').mkdir(exist_ok=True)

    summary = OUTPUT_DIR / 'summary.csv'
    rows = {}
    if summary.exists():
        with summary.open(newline='') as f:
            for row in csv.DictReader(f):
                rows[(row['utt_id'], row['condition'])] = row

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; utterances: {len(paths)}', flush=True)
    synth = PitchFlowerSynthesizer.from_pretrained('diegotg343/PitchFlower').to(device).eval()
    vocos = Vocos.from_pretrained('charactr/vocos-mel-24khz').to(device).eval()

    for i, path in enumerate(paths, 1):
        utt = path.stem
        print(f'[{i}/{len(paths)}] {utt}', flush=True)
        original_path = OUTPUT_DIR / 'original' / f'{utt}.wav'
        pending = []
        for name, alpha, beta in PITCH_CONFIGS:
            out = OUTPUT_DIR / name / f'{utt}.wav'
            contour = OUTPUT_DIR / 'contours' / f'{utt}_{name}.npz'
            plot = OUTPUT_DIR / 'plots' / f'{utt}_{name}.png'
            if ((utt, name) not in rows or not out.is_file() or not contour.is_file()
                    or not original_path.is_file() or (save_plots and not plot.is_file())):
                pending.append((name, alpha, beta, out, contour, plot))
        if not pending:
            print('  Already complete', flush=True)
            continue

        wav, sr = torchaudio.load(str(path))
        wav = wav.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        torchaudio.save(str(original_path), wav, SAMPLE_RATE)
        source, time = extract_f0(wav)
        source_stats = f0_stats(source, 'source')

        # Reuse the same model calls and frame alignment as the existing script.
        with torch.inference_mode():
            codes = synth.analyze_audios(wav.to(device))
            features = synth.autoencoder.quantizer.decode(codes)
            features = synth.autoencoder.quantizer_out(features)
            rec_mel = synth.autoencoder.decoder(features)
            n_frames = rec_mel.shape[-1]
            del features, rec_mel

        for name, alpha, beta, out, contour, plot in pending:
            print(f'  {name}', flush=True)
            target_f0 = modify_f0(source, alpha, beta)
            # Record stats of the original requested contour, not padded frames.
            y = np.zeros_like(target_f0)
            voiced = target_f0 > 0
            y[voiced] = np.log2(target_f0[voiced])
            y = y[:n_frames]
            if len(y) < n_frames:
                y = np.pad(y, (0, n_frames - len(y)))
            target_tensor = torch.tensor(y, dtype=torch.float32, device=device).unsqueeze(0)
            torch.manual_seed(SEED)
            with torch.inference_mode():
                mel = synth.synthesize_mels(codes, target_tensor, steps=STEPS)
                edited = vocos.decode(mel).cpu().reshape(1, -1)
                edited = edited[:, :wav.shape[-1]]
            torchaudio.save(str(out), edited, SAMPLE_RATE)
            # Measure the saved audio (including any WAV quantization/clipping).
            saved, _ = torchaudio.load(str(out))
            generated, gen_time = extract_f0(saved)

            target_stats = f0_stats(target_f0, 'target')
            gen_stats = f0_stats(generated, 'generated')
            row = {'utt_id': utt, 'speaker_id': utt.split('-')[0],
                   'condition': name, 'alpha': alpha, 'beta': beta,
                   'source_path': str(path), 'output_path': str(out),
                   'duration_seconds': wav.shape[-1] / SAMPLE_RATE,
                   **source_stats, **target_stats, **gen_stats,
                   'requested_mean_shift_semitones': 12 * (target_stats['target_mean_log2'] - source_stats['source_mean_log2']),
                   'achieved_mean_shift_semitones': 12 * (gen_stats['generated_mean_log2'] - source_stats['source_mean_log2']),
                   'mean_error_semitones': 12 * (gen_stats['generated_mean_log2'] - target_stats['target_mean_log2']),
                   'std_error_semitones': gen_stats['generated_std_semitones'] - target_stats['target_std_semitones'],
                   **contour_error(target_f0, generated)}
            np.savez_compressed(contour, source_time=time, source_f0=source,
                                target_time=time, target_f0=target_f0,
                                generated_time=gen_time, generated_f0=generated,
                                conditioning_log2_f0=y)
            if save_plots:
                plot_f0(plot, utt, name, source, time, target_f0, generated, gen_time)
            rows[(utt, name)] = row
            write_summary(summary, rows)
            del mel, target_tensor
        del codes

    print(f'Inference complete. Summary: {summary}', flush=True)


if __name__ == '__main__':
    main()
