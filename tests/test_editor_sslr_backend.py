"""Lightweight editor adapter tests: no model downloads."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def load_backend():
    path = Path(__file__).resolve().parents[1] / "examples/prosody_editor/sslr_backend.py"
    spec = importlib.util.spec_from_file_location("editor_sslr_backend", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_rejects_unconfigured_model(monkeypatch):
    backend = load_backend()
    monkeypatch.delenv("SSLR_CHECKPOINT", raising=False)
    monkeypatch.delenv("F0_PROBE_CHECKPOINT", raising=False)
    with pytest.raises(ValueError, match="SSLR_CHECKPOINT"):
        backend.synthesize_sslr({})


def test_rejects_invalid_target_before_model_load(monkeypatch):
    backend = load_backend()
    monkeypatch.setenv("SSLR_CHECKPOINT", "/tmp/sslr")
    monkeypatch.setenv("F0_PROBE_CHECKPOINT", "/tmp/probe")
    payload = {
        "sample_rate": 16000,
        "audio": np.zeros(1600).tolist(),
        "target_f0": [120.0, -10.0],
    }
    with pytest.raises(ValueError, match="Voiced F0"):
        backend.synthesize_sslr(payload)
