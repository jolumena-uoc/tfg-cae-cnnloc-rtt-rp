"""Smoke tests for the data loading and 4xN matrix construction.

These tests do not require pytest - they can be run as a simple script:
    python tests/test_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from cnnloc_rtt.data import (
    SUBSET_KEYS,
    build_4xN_for_all_subsets,
    build_classic_subsets,
    load_full_dataset,
)


def main() -> int:
    print("[1/4] Loading the full dataset...")
    ds = load_full_dataset()
    assert len(ds.aps) == 4, f"expected 4 APs, got {len(ds.aps)}"
    assert len(ds.ref_locations) == 20, (
        f"expected 20 ref locations, got {len(ds.ref_locations)}"
    )
    print(f"     APs: {len(ds.aps)} | reference points: {len(ds.ref_locations)}")

    print("[2/4] Building classic (single-snapshot) subsets...")
    classic = build_classic_subsets(ds, window_size=None)
    for k, (X, y) in classic.items():
        print(f"     {k:25s} -> X={X.shape}, y={y.shape}")
        assert X.ndim == 2 and X.shape[1] == 4, (
            f"unexpected classic X shape: {X.shape}"
        )
        assert X.shape[0] == y.shape[0]
    assert set(classic.keys()) == set(SUBSET_KEYS), (
        f"unexpected subset keys: {set(classic.keys())}"
    )

    print("[3/4] Building 4xN fingerprint matrices (N=16)...")
    matrices = build_4xN_for_all_subsets(ds, n_window=16, stride=1)
    for k, (X, Y) in matrices.items():
        print(f"     {k:25s} -> X={X.shape}, Y={Y.shape}")
        assert X.ndim == 4 and X.shape[1:3] == (4, 16) and X.shape[-1] == 1, (
            f"unexpected 4xN X shape: {X.shape}"
        )
        assert Y.shape == (X.shape[0], 2)
    print("     all four subsets build cleanly")

    print("[4/4] Sanity checks on numerical values...")
    sample_key = list(matrices.keys())[0]
    X, Y = matrices[sample_key]
    rng = np.random.default_rng(0)
    idx = rng.integers(0, X.shape[0])
    sample = X[idx, :, :, 0]
    print(f"     sample matrix shape: {sample.shape}, "
          f"range=[{sample.min():.3f}, {sample.max():.3f}] (m)")
    assert np.isfinite(sample).all(), "found non-finite values"
    assert sample.min() > -2.0 and sample.max() < 50.0, (
        f"distance values look off: {sample.min()} - {sample.max()}"
    )
    print(f"     coord example: ({Y[idx, 0]:.3f}, {Y[idx, 1]:.3f}) m")

    print("\nAll smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
