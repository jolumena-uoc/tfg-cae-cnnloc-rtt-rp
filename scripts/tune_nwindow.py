"""Hyperparameter scan: sensitivity to the temporal window N.

Trains CAE-CNNLoc 2D-Temporal on `POCO#STANDING` for n_window in
{8, 16, 32} and seeds in {0, 1, 2}, evaluating on the four subsets
(self + cross-device/cross-pose). Persists per-sample errors and a
compact summary in `results/`.

Run from the project root with:
    python scripts/tune_nwindow.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from cnnloc_rtt.data import (
    build_4xN_for_all_subsets,
    load_full_dataset,
    normalize_train_test,
)
from cnnloc_rtt.eval import aggregate_results, evaluate_on_subsets
from cnnloc_rtt.models import CAECNNLocConfig
from cnnloc_rtt.train import TrainConfig, train_one_seed

RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

N_WINDOWS = [8, 16, 32]
SEEDS = [0, 1, 2]
TRAIN_KEY = "POCO#STANDING"
TRAIN_CFG = TrainConfig(
    pretrain_epochs=40,
    finetune_epochs=60,
    batch_size=64,
    val_split=0.2,
    early_stopping_patience=10,
    verbose=0,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    log("Loading dataset ...")
    ds = load_full_dataset()

    all_rows = []
    t_start = time.time()

    for n_window in N_WINDOWS:
        log(f"=== Building 4xN matrices with n_window={n_window} ===")
        matrices = build_4xN_for_all_subsets(ds, n_window=n_window, stride=1)
        for k, (X, Y) in matrices.items():
            log(f"   {k:<22s}: X={X.shape}")

        X_train, Y_train = matrices[TRAIN_KEY]
        others = {k: v[0] for k, v in matrices.items() if k != TRAIN_KEY}
        X_train_n, others_n, norm = normalize_train_test(X_train, others)
        log(f"   Normalisation: mean={norm['mean']:.3f}, std={norm['std']:.3f}")
        eval_subsets = {TRAIN_KEY: (X_train_n, Y_train)}
        eval_subsets.update({k: (others_n[k], matrices[k][1]) for k in others_n})

        cfg = CAECNNLocConfig(n_aps=4, n_window=n_window)
        for seed in SEEDS:
            t0 = time.time()
            regressor, _, history = train_one_seed(
                X_train_n, Y_train, seed, cfg, TRAIN_CFG,
                log_callback=lambda m: log("     " + m),
            )
            elapsed = time.time() - t0
            log(
                f"   seed={seed} done in {elapsed:.1f}s | "
                f"pretrain val_loss min={min(history.pretrain_val_loss):.3f} | "
                f"finetune val_loss min={min(history.finetune_val_loss):.3f}"
            )
            df_seed = evaluate_on_subsets(
                regressor, eval_subsets, TRAIN_KEY,
                method_name=f"CAE-CNNLoc-2D-N{n_window}", seed=seed,
            )
            df_seed["n_window"] = n_window
            df_seed["pretrain_min_val_loss"] = float(min(history.pretrain_val_loss))
            df_seed["finetune_min_val_loss"] = float(min(history.finetune_val_loss))
            df_seed["train_elapsed_s"] = float(elapsed)
            all_rows.append(df_seed)

    total = time.time() - t_start
    log(f"=== Scan done in {total/60:.1f} min ===")

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(RESULTS / "tune_nwindow_results.csv", index=False)
    log(f"Saved {len(df)} rows to results/tune_nwindow_results.csv")

    summary = (
        df.groupby(["n_window", "test_set"])["error"]
        .agg(
            mean_error="mean",
            std_error="std",
            median_error="median",
            p75=lambda s: float(np.percentile(s, 75)),
            p95=lambda s: float(np.percentile(s, 95)),
            n_samples="count",
        )
        .round(3)
        .reset_index()
    )
    summary.to_csv(RESULTS / "tune_nwindow_summary.csv", index=False)

    log("\nPer-subset summary (mean error in metres):\n")
    pivot = (
        summary.pivot(index="test_set", columns="n_window", values="mean_error")
        .reindex(["POCO#STANDING", "POCO#TRIPOD", "SAMSUNG#STANDING", "SAMSUNG#TRIPOD"])
    )
    print(pivot.to_string())

    log("\nCross-device + cross-pose mean error per N (excludes self-eval):\n")
    cross = summary[summary["test_set"] != TRAIN_KEY]
    cross_summary = (
        cross.groupby("n_window")["mean_error"].mean().round(3).to_frame("cross_mean_m")
    )
    print(cross_summary.to_string())
    best_n = int(cross_summary["cross_mean_m"].idxmin())
    log(f"\nBest n_window (lowest cross-device/cross-pose mean error): {best_n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
