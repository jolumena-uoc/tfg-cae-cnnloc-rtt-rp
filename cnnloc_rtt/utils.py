"""Reproducibility helpers and metric utilities."""
from __future__ import annotations

import os
import random
from typing import Iterable

import numpy as np


def set_global_seed(seed: int) -> None:
    """Fix the Python, NumPy and TensorFlow PRNG states.

    TensorFlow is imported lazily so that this module remains lightweight
    when only NumPy-based utilities are needed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf  # noqa: WPS433 (intentional lazy import)
    except Exception:  # noqa: BLE001
        tf = None
    if tf is not None:
        tf.random.set_seed(seed)
        try:
            tf.keras.utils.set_random_seed(seed)
        except Exception:  # noqa: BLE001
            pass


def euclidean_error(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Per-sample Euclidean error (assumes shape (N, 2))."""
    return np.linalg.norm(pred - target, axis=1)


def summary_stats(errors: np.ndarray) -> dict:
    """Return mean, std, median and percentile statistics for an error array."""
    arr = np.asarray(errors, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def aggregate_seed_stats(per_seed: Iterable[dict]) -> dict:
    """Aggregate the per-seed summary stats reporting mean and std across seeds."""
    rows = list(per_seed)
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if k != "n"]
    return {
        "n_seeds": len(rows),
        **{f"{k}_mean": float(np.mean([r[k] for r in rows])) for k in keys},
        **{f"{k}_std": float(np.std([r[k] for r in rows])) for k in keys},
    }
