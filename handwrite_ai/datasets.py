"""MNIST and UCI PenDigits dataset helpers."""
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve
import numpy as np
from .config import DATA, SEED


@dataclass
class DigitSplits:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray


def load_mnist(train_limit: int | None = None, test_limit: int | None = None,
               validation_size: int = 5000, seed: int = SEED) -> DigitSplits:
    """Separate validation from MNIST training; preserve an untouched test set."""
    import tensorflow as tf
    try:
        (x, y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
    except Exception as exc:
        raise RuntimeError(
            "MNIST download failed. Connect to the internet for the first run, "
            "or copy mnist.npz into the local ~/.keras/datasets directory."
        ) from exc
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(x))
    val_ids = order[:validation_size]
    train_ids = order[validation_size:]
    if train_limit is not None:
        train_ids = train_ids[:train_limit]
    if test_limit is not None:
        test_x = test_x[:test_limit]
        test_y = test_y[:test_limit]
    return DigitSplits(
        (x[train_ids].astype("float32") / 255.0)[..., None], y[train_ids],
        (x[val_ids].astype("float32") / 255.0)[..., None], y[val_ids],
        (test_x.astype("float32") / 255.0)[..., None], test_y,
    )


PENDIGITS_BASE = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits"


def load_pendigits(folder: str | Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load UCI PenDigits train/test; 8 ordered xy pairs (spatially resampled)."""
    dest = Path(folder) if folder else DATA / "pendigits"
    dest.mkdir(parents=True, exist_ok=True)
    files = []
    for name in ("pendigits.tra", "pendigits.tes"):
        path = dest / name
        if not path.is_file():
            try:
                urlretrieve(f"{PENDIGITS_BASE}/{name}", path)
            except Exception as exc:
                raise RuntimeError(
                    f"Missing {path}. Download pendigits.tra and pendigits.tes "
                    f"from {PENDIGITS_BASE} and place them in {dest}."
                ) from exc
        files.append(np.loadtxt(path, delimiter=",", dtype=np.float32))
    if any(a.ndim != 2 or a.shape[1] != 17 for a in files):
        raise ValueError("PenDigits file must contain 16 coordinates and one class per row")
    return tuple(a[:, :16].reshape(-1, 8, 2) / 100.0 for a in files)


def sliding_windows(trajectories: np.ndarray, length: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Predict a next xy point using a preceding window within each digit."""
    if trajectories.ndim != 3 or trajectories.shape[2] != 2:
        raise ValueError("Expected trajectories in (samples, steps, 2) format")
    steps = trajectories.shape[1]
    if not (1 <= length < steps):
        raise ValueError("Window length must be smaller than trajectory length")
    x = np.concatenate([trajectories[:, idx:idx+length, :]
                        for idx in range(steps - length)], axis=0)
    y = np.concatenate([trajectories[:, idx+length, :]
                        for idx in range(steps - length)], axis=0)
    return x.astype("float32"), y.astype("float32")
