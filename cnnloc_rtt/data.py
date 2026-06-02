"""Data loading and 4xN fingerprint construction for CAE-CNNLoc 2D-Temporal.

This module is intentionally thin: it reuses the data loading utilities
from the original reproducibility package by Matey-Sanz et al. (2025)
and adds the construction of 4xN AP-time fingerprint matrices needed by
CAE-CNNLoc 2D-Temporal.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths and bridge to the original reproducibility package
# ---------------------------------------------------------------------------
DEFAULT_REPRO_PATH = Path(r"C:\TFG\Comparative_Analysis_Android_RTT_main")


def _ensure_repro_on_path(repro_path: Path = DEFAULT_REPRO_PATH) -> Path:
    """Prepend the original repro package to sys.path so `lib.*` is importable.

    Returns the resolved absolute path. Raises FileNotFoundError if the
    package is missing (the user must keep the dataset side by side).
    """
    repro_path = Path(repro_path).resolve()
    if not repro_path.exists():
        raise FileNotFoundError(
            f"Original reproducibility package not found at {repro_path}. "
            "Please place it next to the codigo/ folder or pass an explicit path."
        )
    if str(repro_path) not in sys.path:
        sys.path.insert(0, str(repro_path))
    return repro_path


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
SUBSET_KEYS = (
    "POCO#STANDING",
    "POCO#TRIPOD",
    "SAMSUNG#STANDING",
    "SAMSUNG#TRIPOD",
)
N_APS = 4
DISTANCE_MM_TO_M = 1e-3


@dataclass(frozen=True)
class Dataset:
    """Container with all measurements and metadata."""

    aps: dict
    ref_locations: dict
    raw: dict  # {Device: {MeasuringType: {point_id: DataFrame}}}


def load_full_dataset(repro_path: Path = DEFAULT_REPRO_PATH) -> Dataset:
    """Load the four subsets, AP positions and reference locations."""
    repro_path = _ensure_repro_on_path(repro_path)
    from lib.data_loading import load_aps, load_files, load_locations
    from lib.model import Device, MeasuringType

    aps = load_aps(os.path.join(repro_path, "01_DATA", "aps.csv"))
    ref_locations = load_locations(os.path.join(repro_path, "01_DATA", "locations.csv"))

    raw = {device: {} for device in (Device.POCO, Device.SAMSUNG)}
    devices = {
        Device.POCO: "poco_f2pro",
        Device.SAMSUNG: "samsung_s24ultra",
    }
    for device, folder in devices.items():
        device_path = os.path.join(repro_path, "01_DATA", folder)
        for mt in (MeasuringType.STANDING, MeasuringType.TRIPOD):
            raw[device][mt] = load_files(device_path, mt)
    return Dataset(aps=aps, ref_locations=ref_locations, raw=raw)


def build_classic_subsets(
    dataset: Dataset, window_size: str | None = None
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Build the classic (single-snapshot) (X, y) subsets used by the baselines.

    Returns a dict with the four subset keys "DEVICE#TYPE" -> (X, y) where
    X is (N, 4) RTT distances in millimetres (as in the original notebook)
    and y is the array of reference point ids.
    """
    _ensure_repro_on_path()
    from lib.data_processing import build_datasets

    sets = build_datasets(dataset.raw, window_size=window_size)
    return {str(k): (np.asarray(X), np.asarray(y)) for k, (X, y) in sets.items()}


# ---------------------------------------------------------------------------
# 4xN AP-time fingerprint construction
# ---------------------------------------------------------------------------

def build_4xN_matrices_for_subset(
    dataset: Dataset,
    device,
    measuring_type,
    n_window: int = 16,
    stride: int = 1,
    distance_to_meters: bool = True,
    window_size: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build the (M, 4, N) fingerprint matrices and the (M, 2) target coords.

    The construction reuses the data sanitisation logic of
    ``lib.data_processing.build_datasets`` from the original
    reproducibility package, which already drops measurements where any
    AP is missing and produces a clean per-reference-point sequence of
    valid (4,) RTT vectors. Sliding windows of length ``n_window`` are
    then built on top of those sequences for each reference point.

    Parameters
    ----------
    dataset
        The Dataset returned by ``load_full_dataset``.
    device, measuring_type
        Enum values from ``lib.model.Device`` and ``lib.model.MeasuringType``.
    n_window
        Number of consecutive RTT measurements per AP that compose each
        fingerprint matrix (the temporal dimension N).
    stride
        Step (in samples) between consecutive overlapping windows.
    distance_to_meters
        If True, divide RTT distances by 1000 (the dataset stores mm).
    window_size
        Optional pandas-style frequency string (e.g. ``"1S"``, ``"5S"``)
        passed to the original ``build_datasets`` to apply a temporal
        median aggregation before constructing the 4xN windows.

    Returns
    -------
    X : np.ndarray of shape (n_samples, 4, n_window, 1)
        Fingerprint matrices ready for Keras Conv2D layers.
    Y : np.ndarray of shape (n_samples, 2)
        Ground-truth (x, y) coordinates of the reference point each
        matrix belongs to.
    """
    _ensure_repro_on_path()
    from lib.data_processing import build_datasets

    # We build the dataset for ONLY this (device, measuring_type) pair.
    one_device_subset = {device: {measuring_type: dataset.raw[device][measuring_type]}}
    sets = build_datasets(one_device_subset, window_size=window_size)
    set_key = f"{device}#{measuring_type}"
    if set_key not in sets:
        raise RuntimeError(f"build_datasets did not return key {set_key}")
    X_clean, y_clean = sets[set_key]
    X_clean = np.asarray(X_clean)
    y_clean = np.asarray(y_clean)
    if distance_to_meters:
        X_clean = X_clean.astype(np.float32) * DISTANCE_MM_TO_M

    matrices: List[np.ndarray] = []
    coords: List[Tuple[float, float]] = []

    # For each reference point, take its consecutive valid RTT vectors and
    # slide windows of length n_window across them.
    for point_id in np.unique(y_clean):
        mask = y_clean == point_id
        seq = X_clean[mask]  # (T_p, 4)
        if len(seq) < n_window:
            continue
        ref = dataset.ref_locations[point_id]
        for start in range(0, len(seq) - n_window + 1, stride):
            window = seq[start:start + n_window, :].T  # (4, n_window)
            matrices.append(window)
            coords.append((ref.x, ref.y))

    if not matrices:
        raise RuntimeError(
            f"No valid windows produced for {device}/{measuring_type}. "
            "Check n_window or the input data."
        )

    X = np.stack(matrices, axis=0).astype(np.float32)
    X = X[..., np.newaxis]  # (n_samples, 4, n_window, 1)
    Y = np.asarray(coords, dtype=np.float32)
    return X, Y


def build_4xN_for_all_subsets(
    dataset: Dataset,
    n_window: int = 16,
    stride: int = 1,
    window_size: str | None = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Convenience wrapper to build matrices for the 4 subsets.

    Returns a dict keyed by ``Device#MeasuringType`` strings.
    """
    _ensure_repro_on_path()
    from lib.model import Device, MeasuringType

    out = {}
    for dev in (Device.POCO, Device.SAMSUNG):
        for mt in (MeasuringType.STANDING, MeasuringType.TRIPOD):
            X, Y = build_4xN_matrices_for_subset(
                dataset,
                dev,
                mt,
                n_window=n_window,
                stride=stride,
                window_size=window_size,
            )
            out[f"{dev}#{mt}"] = (X, Y)
    return out


def normalize_train_test(
    X_train: np.ndarray,
    X_others: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Dict[str, float]]:
    """Standardise inputs using statistics computed only on the training set.

    Returns the normalised training set, a dict with the normalised
    versions of the other subsets and a dict with `mean` and `std`.
    """
    mean = float(X_train.mean())
    std = float(X_train.std()) if X_train.std() > 1e-8 else 1.0
    X_train_n = (X_train - mean) / std
    X_others_n = {k: (v - mean) / std for k, v in X_others.items()}
    return X_train_n, X_others_n, {"mean": mean, "std": std}
