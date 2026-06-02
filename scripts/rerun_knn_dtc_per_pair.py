"""Re-run KNN-DtC labelling each row with the explicit (train, test) pair.

The original `lib.methods.fingerprinting` evaluates a model trained on
one subset against the union of the other three subsets and returns a
single dataframe that does not expose which test set each row came
from. This script wraps that logic to produce a per-pair dataframe so
the comparison against CAE-CNNLoc (which is per-pair) is rigorous.

Run from the project root with:
    python scripts/rerun_knn_dtc_per_pair.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cnnloc_rtt.data import build_classic_subsets, load_full_dataset

# Bring in the reproducibility package so that lib.* is importable.
import importlib  # noqa: E402

from cnnloc_rtt.data import _ensure_repro_on_path  # noqa: E402

_ensure_repro_on_path()

from lib.methods import euclidean_rtt  # noqa: E402  (after sys.path tweak)
from lib.model import AP, Location  # noqa: E402

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

WINDOWS = [None, "1S", "5S"]
WINDOW_LABEL = {None: "raw", "1S": "1s", "5S": "5s"}
K_VALUES = [1, 3, 5, 10]
DISTANCE_MM_TO_M = 1e-3


def knn_dtc_pair(
    train_x: np.ndarray, train_y: np.ndarray,
    test_x: np.ndarray, test_y: np.ndarray,
    ref_locations,
    k: int,
):
    """Run KNN with offset-normalized euclidean distance for a single pair.

    `train_x`/`test_x` are (N, 4) arrays of RTT distances in millimetres.
    Returns the per-sample squared error in metres.
    """
    errors = np.empty(len(test_x), dtype=np.float64)
    for i in range(len(test_x)):
        d = euclidean_rtt(train_x, test_x[i])
        sorted_idx = np.argsort(d)[:k]
        closest = train_y[sorted_idx]
        coords = np.array(
            [[ref_locations[p].x, ref_locations[p].y] for p in closest]
        )
        pred = coords.mean(axis=0)
        gt = ref_locations[test_y[i]]
        errors[i] = float(np.linalg.norm(pred - np.array([gt.x, gt.y])))
    return errors


def main() -> int:
    t0 = time.time()
    ds = load_full_dataset()

    rows = []
    for w in WINDOWS:
        sets = build_classic_subsets(ds, window_size=w)
        # convert numpy arrays for use
        sets = {k: (np.asarray(X), np.asarray(y)) for k, (X, y) in sets.items()}
        for train_key, (X_tr, y_tr) in sets.items():
            for test_key, (X_te, y_te) in sets.items():
                for k in K_VALUES:
                    if train_key == test_key and k > 1:
                        # k>1 with self-eval is meaningless: skip
                        continue
                    errors = knn_dtc_pair(X_tr, y_tr, X_te, y_te,
                                           ds.ref_locations, k=k)
                    df = pd.DataFrame({
                        "method": f"KNN-DtC-k{k}",
                        "window": WINDOW_LABEL[w],
                        "train_set": train_key,
                        "test_set": test_key,
                        "k": k,
                        "error": errors,
                    })
                    rows.append(df)
                    print(
                        f"  KNN-DtC k={k:<2d} | window={WINDOW_LABEL[w]:<3s} | "
                        f"train={train_key:<22s} test={test_key:<22s} | "
                        f"mean={errors.mean():.3f} m  median={np.median(errors):.3f} m"
                    )
    df_all = pd.concat(rows, ignore_index=True)
    df_all.to_csv(RESULTS / "01_baselines_knn_dtc_per_pair.csv", index=False)
    print(
        f"\nSaved {len(df_all)} rows to results/01_baselines_knn_dtc_per_pair.csv "
        f"in {(time.time()-t0)/60:.1f} min"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
