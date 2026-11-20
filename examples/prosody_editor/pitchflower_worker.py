"""Persistent line-oriented PitchFlower worker in its own PyTorch 2.8 venv."""
import base64
from contextlib import redirect_stdout
import json
import sys
import traceback

from pitchflower_backend import _synthesize_in_worker


def main():
    protocol = sys.stdout
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            # Reserve stdout for the JSON protocol; upstream code may print.
            with redirect_stdout(sys.stderr):
                wav = _synthesize_in_worker(payload)
            reply = {"wav": base64.b64encode(wav).decode("ascii")}
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            reply = {"error": f"{type(exc).__name__}: {exc}"}
        protocol.write(json.dumps(reply) + "\n")
        protocol.flush()


if __name__ == "__main__":
    main()
