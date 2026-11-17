"""Safe and lightweight experiment artifact persistence."""
from pathlib import Path
from datetime import datetime, timezone
import json
import numpy as np
from .config import ARTIFACTS, ensure_dirs


def save_experiment(name: str, metrics: dict, history: dict | None = None) -> None:
    ensure_dirs()
    record = {
        "experiment": name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "history": history or {},
    }
    (ARTIFACTS / f"{name}.json").write_text(
        json.dumps(record, indent=2, default=_encode), encoding="utf-8"
    )


def _encode(value):
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value)}")


def load_experiment(name: str) -> dict | None:
    path = ARTIFACTS / f"{name}.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def saved_experiments() -> dict[str, dict]:
    ensure_dirs()
    result = {}
    for path in sorted(ARTIFACTS.glob("*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            if "metrics" in record:
                result[path.stem] = record
        except (ValueError, OSError):
            continue
    return result
