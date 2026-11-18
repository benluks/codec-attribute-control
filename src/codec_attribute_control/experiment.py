"""FACodec frozen prosody probe using quick-convert's dataset/F0 APIs."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from quick_convert.data import BaseDataset
from quick_convert.utils.audio import load_audio as qc_load_audio

from .facodec import FACodec, HOP, SAMPLE_RATE
from .f0 import aligned_pitch, make_extractor
from .targets import TARGET_ID, f0_target, ridge_intervention, semitone_delta_z


def load_wave(path, *, sample_rate):
    wave, sr = qc_load_audio(path, target_sr=sample_rate, mono=True)
    if sr != sample_rate or wave.shape[0] != 1:
        raise ValueError("Expected loaded mono audio at target sample rate")
    return wave[0]


def label_pitch(path, extractor, frame_count):
    sr = getattr(extractor, "sample_rate", SAMPLE_RATE)
    wave = load_wave(path, sample_rate=sr)
    return aligned_pitch(extractor, wave, sr, frame_count)


def train(root, output, *, backend="jdc", alpha=100.0, seed=42, max_recordings=50,
          checkpoint_path=None, device="cpu"):
    """Train a simple recording-disjoint linear probe."""
    codec = FACodec(device=device)
    extractor = make_extractor(backend, device=device, checkpoint_path=checkpoint_path)
    dataset = BaseDataset(
        root=root, file_format=["wav", "flac"], utt_id_template="{path.stem}",
        load=["audio"], target_sr=SAMPLE_RATE, convert_to_mono=True,
    )
    if max_recordings is not None and max_recordings < 3:
        raise ValueError("max_recordings must be >=3")
    xs, ys, groups, used, skipped = [], [], [], [], []
    for idx in range(min(len(dataset), max_recordings or len(dataset))):
        sample = dataset[idx]
        if sample.waveform is None or sample.sample_rate != SAMPLE_RATE:
            raise RuntimeError("quick-convert dataset did not load 16kHz audio")
        frame = codec.encode(sample.waveform[0])
        feature = frame.prosody[0].transpose(0, 1).detach().cpu().numpy()
        pitch_hz = label_pitch(sample.path, extractor, feature.shape[0])
        try:
            target, _ = f0_target(pitch_hz)
        except ValueError:
            skipped.append(sample.utt_id)
            continue
        voiced = target != -10
        xs.append(feature[voiced])
        ys.append(target[voiced])
        groups.extend([len(used)] * int(voiced.sum()))
        used.append(sample.utt_id)
    if len(used) < 3:
        raise ValueError("Need at least three usable voiced recordings")
    X = np.concatenate(xs)
    y = np.concatenate(ys)
    group_ids = np.asarray(groups)
    rng = np.random.default_rng(seed)
    heldout = rng.permutation(len(used))[:max(1, round(len(used) * 0.2))]
    mask = np.isin(group_ids, heldout)
    probe = Ridge(alpha=alpha).fit(X[~mask], y[~mask])
    metrics = {
        "val_mae_normalized_log2_f0": float(mean_absolute_error(y[mask], probe.predict(X[mask]))),
        "train_frames": int((~mask).sum()),
        "val_frames": int(mask.sum()),
        "accepted_utt_ids": used,
        "skipped_utt_ids": skipped,
    }
    artifact = {
        "probe": probe, "target": TARGET_ID, "f0_backend": backend,
        "sample_rate": SAMPLE_RATE, "hop": HOP, "metrics": metrics,
        "f0_checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "normalization_scope": "recording",
    }
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output)
    return metrics


def _pitch_from_wave(waveform, extractor, n_frames):
    import torchaudio
    rate = getattr(extractor, "sample_rate", SAMPLE_RATE)
    y = torch.as_tensor(waveform, dtype=torch.float32)
    if rate != SAMPLE_RATE:
        y = torchaudio.functional.resample(y, SAMPLE_RATE, rate)
    return aligned_pitch(extractor, y, rate, n_frames)


def intervene(audio, probe_path, output_dir, *, shifts=(-6, -3, 0, 3, 6), device="cpu"):
    artifact = joblib.load(probe_path)  # Only load trusted probe artifacts.
    if (artifact.get("target") != TARGET_ID or artifact.get("sample_rate") != SAMPLE_RATE
            or artifact.get("hop") != HOP):
        raise ValueError("Probe/checkpoint representation mismatch")
    codec = FACodec(device=device)
    extractor = make_extractor(
        artifact["f0_backend"], device=device,
        checkpoint_path=artifact.get("f0_checkpoint_path"),
    )
    wave = load_wave(audio, sample_rate=SAMPLE_RATE)
    frame = codec.encode(wave)
    f0 = label_pitch(audio, extractor, frame.prosody.shape[-1])
    _, stats = f0_target(f0)
    coef = np.asarray(artifact["probe"].coef_).reshape(-1)
    if len(coef) != frame.prosody.shape[1]:
        raise ValueError("Probe feature dimension mismatch")
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    n_samples = len(wave)
    baseline = codec.decode(frame)[:n_samples].numpy()
    sf.write(folder / "baseline.wav", baseline, SAMPLE_RATE)
    base_pitch = _pitch_from_wave(baseline, extractor, len(f0))
    results = []
    for semitones in shifts:
        requested = semitone_delta_z(f0, semitones)
        delta_tc = ridge_intervention(coef, requested)
        delta = torch.from_numpy(delta_tc.T[None]).to(
            device=frame.prosody.device, dtype=frame.prosody.dtype,
        )
        modified = codec.decode(frame, delta)[:n_samples].numpy()
        name = f"shift_{semitones:+g}st.wav"
        sf.write(folder / name, modified, SAMPLE_RATE)
        changed_pitch = _pitch_from_wave(modified, extractor, len(f0))
        valid = (np.isfinite(f0) & np.isfinite(base_pitch) & np.isfinite(changed_pitch)
                 & (f0 > 5.0) & (base_pitch > 5.0) & (changed_pitch > 5.0))
        shift = (
            float(np.median(12.0 * np.log2(changed_pitch[valid] / base_pitch[valid])))
            if valid.any() else None
        )
        results.append({
            "requested_semitones": float(semitones),
            "achieved_median_semitones": shift,
            "paired_voiced_frames": int(valid.sum()),
            "audio": name,
        })
    report = {
        "f0_backend": artifact["f0_backend"],
        "source_std_log2_f0": stats.std_log2_hz,
        "results": results,
        "warning": "Fixed-source-stat shifts are exploratory: segment z-scoring "
                   "removes absolute pitch offsets.",
    }
    (folder / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
