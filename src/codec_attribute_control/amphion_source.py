"""Locate the pinned upstream Amphion Git submodule without PYTHONPATH setup."""
from pathlib import Path
import sys


def ensure_amphion_importable() -> Path:
    root = Path(__file__).resolve().parents[2]
    source = root / "external" / "Amphion"
    if not (source / "models" / "codec" / "ns3_codec" / "facodec.py").is_file():
        raise ImportError(
            "Amphion submodule is missing. Run: git submodule update --init"
        )
    parent = str(source.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    return source
