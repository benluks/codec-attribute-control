"""Minimal local WORLD synthesis server for the prosody editor.

Run: uv run --extra facodec python examples/prosody_editor/world_server.py
Access from a laptop via SSH port forwarding, not a public cluster port.
"""
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit
from io import BytesIO

import numpy as np
import soundfile as sf

from sslr_backend import synthesize_sslr

INDEX = Path(__file__).with_name("index.html")
MAX_BYTES = 20 * 1024 * 1024
MAX_SECONDS = 30
SR = 16000
HOP_MS = 12.5


def synthesize(payload):
    import pyworld

    sr = int(payload["sample_rate"])
    if sr < 8000 or sr > 96000:
        raise ValueError("Unsupported input sample rate")
    audio = np.asarray(payload["audio"], dtype=np.float64)
    if audio.ndim != 1 or not 0 < audio.size <= sr * MAX_SECONDS:
        raise ValueError("Expected mono audio of at most 30 seconds")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains nonfinite samples")
    from scipy.signal import resample_poly
    from math import gcd

    g = gcd(sr, SR)
    audio = resample_poly(audio, SR // g, sr // g).astype(np.float64)
    audio = np.ascontiguousarray(audio)
    f0, times = pyworld.dio(audio, SR, frame_period=HOP_MS)
    f0 = pyworld.stonemask(audio, f0, times, SR)
    spectral = pyworld.cheaptrick(audio, f0, times, SR)
    aperiodic = pyworld.d4c(audio, f0, times, SR)
    edits = payload["target_f0"]
    if not isinstance(edits, list) or len(edits) > 10000:
        raise ValueError("Invalid target F0")
    # Null denotes unvoiced. Preserve WORLD's voicing decision: only edit
    # voiced frames, without synthesizing pitch in originally unvoiced regions.
    target = np.asarray([np.nan if v is None else float(v) for v in edits])
    if target.size < 2:
        raise ValueError("F0 contour too short")
    if np.any(np.isfinite(target) & ((target < 0) | (target > 1500))):
        raise ValueError("F0 out of range")
    nearest = np.clip(np.rint(times / 0.0125).astype(int), 0, target.size - 1)
    selected = target[nearest]
    changed = np.where((f0 > 0) & np.isfinite(selected) & (selected > 0), selected, 0.0)
    rendered = pyworld.synthesize(changed, spectral, aperiodic, SR, frame_period=HOP_MS)
    out = BytesIO()
    sf.write(out, np.clip(rendered, -1, 1), SR, format="WAV", subtype="PCM_16")
    return out.getvalue()


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlsplit(self.path).path in ("/", "/index.html"):
            data = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
        elif urlsplit(self.path).path == "/api/health":
            data = json.dumps({"backends": ["world"] + (["w2vbert"] if os.environ.get("SSLR_CHECKPOINT") and os.environ.get("F0_PROBE_CHECKPOINT") else [])}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
        else:
            self.send_error(404)
            return
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/api/convert":
            self.send_error(404)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_BYTES:
                raise ValueError("Request too large")
            payload = json.loads(self.rfile.read(size))
            backend = payload.get('backend', 'world')
            if backend == 'world':
                data = synthesize(payload)
            elif backend == 'w2vbert':
                data = synthesize_sslr(payload)
            else:
                raise ValueError('Unsupported conversion backend')
        except (ValueError, KeyError, TypeError, ImportError) as exc:
            data = json.dumps({"error": str(exc)}).encode()
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
        except Exception as exc:
            data = json.dumps({"error": f"Conversion failed: {exc}"}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
        else:
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    host = os.environ.get("EDITOR_HOST", "127.0.0.1")
    port = int(os.environ.get("EDITOR_PORT", "8765"))
    print(f"WORLD editor: http://{host}:{port}", flush=True)
    ThreadingHTTPServer((host, port), Handler).serve_forever()
