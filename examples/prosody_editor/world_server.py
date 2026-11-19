#!/usr/bin/env python3
"""Local WORLD conversion backend. Run with:
uv run --no-project --with numpy --with pyworld python examples/prosody_editor/world_server.py
"""
import base64
import io
import json
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pyworld as pw


def convert(payload):
    raw = base64.b64decode(payload["audio_wav_base64"], validate=True)
    with wave.open(io.BytesIO(raw), "rb") as f:
        channels, width, fs, n = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        if width != 2 or channels != 1:
            raise ValueError("Expected mono 16-bit PCM WAV")
        x = np.frombuffer(f.readframes(n), dtype="<i2").astype(np.float64) / 32768
    if not 8000 <= fs <= 96000 or not fs // 10 <= len(x) <= fs * 120:
        raise ValueError("Unsupported sample rate or audio duration")
    frames = payload["frames"]
    t = np.array([f["t"] for f in frames], dtype=float)
    src = np.array([np.nan if f["original_hz"] is None else f["original_hz"] for f in frames])
    dst = np.array([np.nan if f["target_hz"] is None else f["target_hz"] for f in frames])
    if len(t) == 0 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Invalid F0 timestamps")
    valid = np.isfinite(src) & np.isfinite(dst) & (src > 0) & (dst > 0)
    delta = np.zeros(len(t))
    delta[valid] = np.clip(12 * np.log2(dst[valid] / src[valid]), -36, 36)

    f0, times = pw.dio(x, fs, frame_period=5.0)
    f0 = pw.stonemask(x, f0, times, fs)
    sp = pw.cheaptrick(x, f0, times, fs)
    ap = pw.d4c(x, f0, times, fs)
    delta_world = np.zeros_like(times)
    edges = np.diff(np.r_[False, valid, False].astype(np.int8))
    for lo, hi in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        mask = (times >= t[lo]) & (times <= t[hi - 1])
        delta_world[mask] = np.interp(times[mask], t[lo:hi], delta[lo:hi])
    voiced = f0 > 0
    new_f0 = f0.copy()
    new_f0[voiced] = np.clip(f0[voiced] * 2 ** (delta_world[voiced] / 12), 50, 1000)
    y = np.nan_to_num(pw.synthesize(new_f0, sp, ap, fs))
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > .98:
        y *= .98 / peak
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(fs)
        w.writeframes((np.clip(y, -1, 1) * 32767).astype("<i2").tobytes())
    return out.getvalue()


class Handler(BaseHTTPRequestHandler):
    def respond(self, code, data, mime="text/plain"):
        self.send_response(code)
        self.send_header("Content-Type", mime)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS, GET")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.respond(204, b"")

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, b'{"backend":"WORLD","ready":true}', "application/json")
        else:
            self.respond(404, b"Not found")

    def do_POST(self):
        if self.path != "/convert":
            return self.respond(404, b"Not found")
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if size <= 0 or size > 30 * 1024 * 1024:
                raise ValueError("Invalid request size")
            result = convert(json.loads(self.rfile.read(size)))
            self.respond(200, result, "audio/wav")
        except (ValueError, KeyError, TypeError) as e:
            self.respond(400, str(e).encode())
        except Exception as e:
            self.respond(500, ("Conversion failed: " + str(e)).encode())


if __name__ == "__main__":
    print("WORLD server at http://127.0.0.1:8765", flush=True)
    ThreadingHTTPServer(("127.0.0.1", 8765), Handler).serve_forever()
