"""PitchFlower editor adapter tests without downloading pretrained weights."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


def backend():
    file = Path(__file__).resolve().parents[1] / "examples/prosody_editor/pitchflower_backend.py"
    spec = importlib.util.spec_from_file_location("editor_pitchflower", file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_edited_contour_resampled_to_mel_clock():
    module = backend()
    contour = np.array([100.0, 200.0, 300.0], dtype=np.float64)
    # Three model frames: 0, 10, 20 ms; browser frames: 0, 20, 40 ms.
    got = module.target_log2_f0(contour, 0.02, 3, 240, 24000)
    np.testing.assert_allclose(got, np.log2([100, 150, 200]), atol=1e-6)


def test_unvoiced_gaps_not_interpolated():
    module = backend()
    contour = np.array([100.0, np.nan, 200.0, 300.0])
    got = module.target_log2_f0(contour, 0.02, 8, 240, 24000)
    assert got[0] == pytest.approx(np.log2(100.0))
    assert np.all(got[1:4] == 0.0)
    assert got[4] == pytest.approx(np.log2(200.0))
    assert got[5] == pytest.approx(np.log2(250.0))
    assert got[6] == pytest.approx(np.log2(300.0))
    assert got[7] == 0.0


def test_validate_input_without_model():
    module = backend()
    payload = {
        "sample_rate": 16000,
        "audio": np.zeros(1600).tolist(),
        "target_f0": [120.0, None, 140.0],
        "target_frame_period": 0.0125,
    }
    sr, audio, target, hop = module._request(payload)
    assert sr == 16000 and len(audio) == 1600
    assert np.isnan(target[1]) and hop == pytest.approx(0.0125)


def test_rejects_negative_pitch():
    module = backend()
    with pytest.raises(ValueError, match="Voiced target"):
        module._request({
            "sample_rate": 16000,
            "audio": [0.0] * 100,
            "target_f0": [200.0, -1.0],
        })


def test_rejects_infinite_contour():
    module = backend()
    with pytest.raises(ValueError, match="Voiced target"):
        module._request({
            "sample_rate": 16000,
            "audio": [0.0] * 100,
            "target_f0": [200.0, float("inf")],
        })


def test_parent_uses_worker_and_preserves_edited_contour(monkeypatch):
    import base64
    import io
    import json

    module = backend()
    expected = b"RIFFtest"
    response = json.dumps({"wav": base64.b64encode(expected).decode()}) + "\n"

    class FakeProcess:
        stdin = io.StringIO()
        stdout = io.StringIO(response)

        def poll(self):
            return None

    fake = FakeProcess()
    monkeypatch.setattr(module, "_WORKER_PROCESS", None)
    monkeypatch.setattr(module, "_start_worker", lambda: fake)
    payload = {
        "sample_rate": 16000,
        "audio": [0.0] * 160,
        "target_f0": [125.0, None, 220.0],
        "target_frame_period": 0.0125,
    }
    assert module.synthesize_pitchflower(payload) == expected
    sent = json.loads(fake.stdin.getvalue().strip())
    assert sent["target_f0"] == payload["target_f0"]
    assert sent["target_frame_period"] == pytest.approx(0.0125)


def test_worker_requires_separate_python(monkeypatch):
    module = backend()
    monkeypatch.delenv("PITCHFLOWER_PYTHON", raising=False)
    with pytest.raises(ValueError, match="PITCHFLOWER_PYTHON"):
        module._start_worker()


def test_pitchflower_worker_preserves_venv_symlink(tmp_path, monkeypatch):
    import subprocess
    import sys

    module = backend()
    interpreter = tmp_path / "python"
    interpreter.symlink_to(sys.executable)
    monkeypatch.setenv("PITCHFLOWER_PYTHON", str(interpreter))
    monkeypatch.setattr(subprocess, "Popen", lambda argv, **kwargs: argv)
    assert module._start_worker()[0] == str(interpreter)
