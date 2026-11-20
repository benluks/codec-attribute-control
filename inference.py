import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
import pyworld as pw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pitchflower.synthesizer import PitchFlowerSynthesizer
from vocos import Vocos

# ==========================================
# Configuration
# ==========================================

SOURCE_DIR = Path(
    "/medias/speech/data/LibriSpeech/LibriSpeech/test-clean"
)

OUTPUT_DIR = Path(
    "/medias/speech/projects/garga/hackathon/pitchflower"
)

NUM_UTTERANCES = 100
SAMPLE_RATE = 24000
SEMITONES = [-6, -3, 0, 3, 6]
STEPS = 10
SEED = 42

SAVE_PLOTS = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", device)

# ==========================================
# 1. Select LibriSpeech utterances
# ==========================================

audio_files = sorted(SOURCE_DIR.rglob("*.flac"))

assert len(audio_files) >= NUM_UTTERANCES, \
    "Insufficient audio files found"

random.Random(SEED).shuffle(audio_files)
audio_files = audio_files[:NUM_UTTERANCES]

print(f"Selected {len(audio_files)} utterances")

# Save selected utterances for reproducibility
with open(OUTPUT_DIR / "selected_utterances.txt", "w") as f:
    for path in audio_files:
        f.write(str(path) + "\n")

# ==========================================
# 2. Load pretrained models once
# ==========================================

synth = PitchFlowerSynthesizer.from_pretrained(
    "diegotg343/PitchFlower"
).to(device).eval()

vocos = Vocos.from_pretrained(
    "charactr/vocos-mel-24khz"
).to(device).eval()

# ==========================================
# 3. F0 extraction
# ==========================================

def extract_f0(wav):

    audio = wav.reshape(-1).cpu().numpy().astype(np.float64)

    f0, time = pw.harvest(
        audio,
        fs=SAMPLE_RATE,
        frame_period=32 / 3,
        f0_floor=20
    )

    f0 = pw.stonemask(
        audio, f0, time, SAMPLE_RATE
    )

    return f0, time


def modify_f0(f0, semitones):

    target_f0 = f0.copy()
    voiced = f0 > 0

    target_f0[voiced] *= 2 ** (semitones / 12)

    return target_f0

# ==========================================
# 4. Create output directories
# ==========================================

(OUTPUT_DIR / "original").mkdir(exist_ok=True)

for shift in SEMITONES:

    (OUTPUT_DIR / f"shift_{shift:+d}").mkdir(
        exist_ok=True
    )

if SAVE_PLOTS:
    (OUTPUT_DIR / "plots").mkdir(exist_ok=True)

# ==========================================
# 5. Inference
# ==========================================

for i, audio_path in enumerate(audio_files):

    utt_id = audio_path.stem

    print(
        f"\n[{i+1}/{len(audio_files)}] {utt_id}",
        flush=True
    )

    # Skip utterances already fully processed
    expected_files = [
        OUTPUT_DIR / f"shift_{s:+d}" / f"{utt_id}.wav"
        for s in SEMITONES
    ]

    expected_files.append(
        OUTPUT_DIR / "original" / f"{utt_id}.wav"
    )

    if SAVE_PLOTS:
        expected_files.extend(
            OUTPUT_DIR / "plots" / f"{utt_id}_shift_{s:+d}.png"
            for s in SEMITONES
        )

    if all(p.exists() for p in expected_files):
        print("Already processed. Skipping.")
        continue

    # Load audio
    wav, sr = torchaudio.load(str(audio_path))
    wav = wav.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(
            wav, sr, SAMPLE_RATE
        )

    # Save original audio at 24 kHz
    torchaudio.save(
        str(OUTPUT_DIR / "original" / f"{utt_id}.wav"),
        wav,
        SAMPLE_RATE
    )

    # Extract F0
    source_f0, time = extract_f0(wav)

    # Encode each utterance only once
    with torch.inference_mode():

        codes = synth.analyze_audios(wav.to(device))

        features = synth.autoencoder.quantizer.decode(codes)
        features = synth.autoencoder.quantizer_out(features)

        rec_mel = synth.autoencoder.decoder(features)

        n_frames = rec_mel.shape[-1]

    # ======================================
    # 6. Generate different pitch shifts
    # ======================================

    for shift in SEMITONES:

        print(f"  Generating {shift:+d} semitones",
              flush=True)

        target_f0 = modify_f0(source_f0, shift)

        target_log_f0 = np.zeros_like(target_f0)
        voiced = target_f0 > 0

        target_log_f0[voiced] = np.log2(
            target_f0[voiced]
        )

        # Align F0 with mel frames
        target_log_f0 = target_log_f0[:n_frames]

        if len(target_log_f0) < n_frames:
            target_log_f0 = np.pad(
                target_log_f0,
                (0, n_frames - len(target_log_f0))
            )

        target = torch.tensor(
            target_log_f0,
            dtype=torch.float32,
            device=device
        ).unsqueeze(0)

        torch.manual_seed(SEED)

        with torch.inference_mode():

            mel = synth.synthesize_mels(
                codes,
                target,
                steps=STEPS
            )

            edited_wav = vocos.decode(mel).cpu()

        edited_wav = edited_wav.reshape(1, -1)
        edited_wav = edited_wav[:, :wav.shape[-1]]

        # Save generated audio
        output_path = (
            OUTPUT_DIR
            / f"shift_{shift:+d}"
            / f"{utt_id}.wav"
        )

        torchaudio.save(
            str(output_path),
            edited_wav,
            SAMPLE_RATE
        )

        # ==================================
        # 7. F0 evaluation and plots
        # ==================================

        if SAVE_PLOTS:

            predicted_f0, predicted_time = extract_f0(
                edited_wav
            )

            plt.figure(figsize=(12, 4))

            plt.plot(
                time, source_f0,
                label="Original F0"
            )

            plt.plot(
                time, target_f0,
                label="Target F0"
            )

            plt.plot(
                predicted_time, predicted_f0,
                label="Generated F0",
                linestyle="--"
            )

            plt.xlabel("Time (s)")
            plt.ylabel("F0 (Hz)")
            plt.title(f"{utt_id}: {shift:+d} semitones")
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                OUTPUT_DIR / "plots"
                / f"{utt_id}_shift_{shift:+d}.png",
                dpi=150
            )

            plt.close()

    del codes, features, rec_mel, mel

print("\nInference complete.")