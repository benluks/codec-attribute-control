"""Frozen FACodec probe and resynthesis experiment."""
from pathlib import Path
import json

import joblib
import numpy as np
import soundfile as sf
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

from .facodec import FACodec, HOP, SAMPLE_RATE
from .f0 import extract_pitch
from .targets import TARGET_ID, f0_target, ridge_intervention, semitone_delta_z


def load_audio(path):
    from quick_convert.utils.audio import load_audio as qc_load_audio
    audio, sr = qc_load_audio(path, target_sr=SAMPLE_RATE, mono=True)
    if sr != SAMPLE_RATE or audio.shape[0] != 1:
        raise ValueError("Expected mono 16 kHz audio")
    return audio[0]


def audio_files(root):
    """File discovery stays local; audio decoding is delegated to quick-convert."""
    root = Path(root)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac"})
    if not paths:
        raise ValueError(f"No wav/flac files under {root}")
    return paths


def train(root, output, *, backend="pyin", alpha=100.0, seed=42, max_recordings=50, device="cpu"):
    codec = FACodec(device=device)
    xs, ys, recording_ids = [], [], []
    accepted, rejected = [], []
    paths = audio_files(root)
    for path in paths[:max_recordings]:
        wave = load_audio(path)
        frame = codec.encode(wave)
        z = frame.prosody[0].transpose(0, 1).cpu().numpy()
        f0 = extract_pitch(wave, len(z), backend)
        try:
            target, _ = f0_target(f0)
        except ValueError:
            rejected.append(str(path))
            continue
        mask = target != -10
        xs.append(z[mask])
        ys.append(target[mask])
        recording_ids.extend([len(accepted)] * int(mask.sum()))
        accepted.append(str(path))
    if len(accepted) < 3:
        raise ValueError("Need at least three voiced recordings")
    X, y = np.concatenate(xs), np.concatenate(ys)
    groups = np.asarray(recording_ids)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(accepted))
    heldout = order[:max(1, round(len(accepted) * 0.2))]
    val = np.isin(groups, heldout)
    probe = Ridge(alpha=alpha).fit(X[~val], y[~val])
    metrics = {
        "mae_normalized_log2_f0": float(mean_absolute_error(y[val], probe.predict(X[val]))),
        "train_frames": int((~val).sum()), "validation_frames": int(val.sum()),
        "accepted_recordings": accepted, "rejected_recordings": rejected,
    }
    result = {"probe": probe, "target": TARGET_ID, "backend": backend,
              "sample_rate": SAMPLE_RATE, "hop": HOP, "metrics": metrics}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(result, output)
    return metrics


def intervene(audio, probe_path, output_dir, *, shifts=(-6, -3, 0, 3, 6), device="cpu"):
    artifact = joblib.load(probe_path)  # Only load artifacts you trust.
    if (artifact.get("target") != TARGET_ID or artifact.get("sample_rate") != SAMPLE_RATE
            or artifact.get("hop") != HOP):
        raise ValueError("Incompatible probe target or codec frame rate")
    codec = FACodec(device=device)
    wave = load_audio(audio)
    frame = codec.encode(wave)
    f0 = extract_pitch(wave, frame.prosody.shape[-1], artifact["backend"])
    _, stats = f0_target(f0)
    coef = np.asarray(artifact["probe"].coef_).reshape(-1)
    if len(coef) != frame.prosody.shape[1]:
        raise ValueError("Probe feature dimension differs from prosody embedding")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    original_len = wave.numel()
    baseline = codec.decode(frame)[:original_len].numpy()
    sf.write(output_dir / "baseline.wav", baseline, SAMPLE_RATE)
    base_pitch = extract_pitch(torch.from_numpy(baseline), len(f0), artifact["backend"])
    rows = []
    for semitones in shifts:
        delta_z = semitone_delta_z(f0, semitones)
        delta = ridge_intervention(coef, delta_z)
        delta = torch.from_numpy(delta.T[None]).to(frame.prosody.device, frame.prosody.dtype)
        modified = codec.decode(frame, delta)[:original_len].numpy()
        name = f"shift_{semitones:+g}st.wav"
        sf.write(output_dir / name, modified, SAMPLE_RATE)
        modified_pitch = extract_pitch(torch.from_numpy(modified), len(f0), artifact["backend"])
        paired = np.isfinite(base_pitch) & np.isfinite(modified_pitch) & np.isfinite(f0)
        paired &= (base_pitch > 5) & (modified_pitch > 5) & (f0 > 5)
        achieved = float(np.median(12 * np.log2(modified_pitch[paired] / base_pitch[paired]))) if paired.any() else None
        rows.append({"requested_semitones": semitones, "achieved_median_semitones": achieved,
                     "paired_voiced_frames": int(paired.sum()), "audio": name})
    report = {"source_std_log2_f0": stats.std_log2_hz, "results": rows,
              "note": "Fixed-source-stat pitch offsets are exploratory: per-segment normalization removes absolute shifts."}
    (output_dir / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
