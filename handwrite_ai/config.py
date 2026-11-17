"""File locations and reproducible defaults for HandWrite AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
ASSETS = ROOT / "assets"
SEED = 42


def ensure_dirs() -> None:
    for path in (ARTIFACTS, DATA, ASSETS):
        path.mkdir(parents=True, exist_ok=True)
