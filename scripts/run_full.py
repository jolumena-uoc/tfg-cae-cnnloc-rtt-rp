"""Full-scale CAE-CNNLoc 2D-Temporal experiment.

Runs the experiment with 5 seeds x 4 training subsets, evaluating each
trained model on the four subsets (self + cross-device/cross-pose).

The hyperparameter `n_window` is read from the command line; pick the
best value reported by `scripts/tune_nwindow.py`.

Run from the project root with:
    python scripts/run_full.py --n-window 16
"""
from __future__ import annotations

import argparse
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

SEEDS = [0, 1, 2, 3, 4]
TRAIN_KEYS = ["POCO#STANDING", "POCO#TRIPOD", "SAMSUNG#STANDING", "SAMSUNG#TRIPOD"]
TRAIN_CFG = TrainConfig(
    pretrain_epochs=60,
    finetune_epochs=80,
    batch_size=64,
    val_split=0.2,
    early_stopping_patience=12,
    verbose=0,
)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--n-window", type=int, default=16, help="temporal window size N")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS, help="random seeds")
    p.add_argument(
        "--train-keys",
        type=str,
        nargs="+",
        default=TRAIN_KEYS,
        help="training subset keys",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    n_window = int(args.n_window)
    seeds = list(args.seeds)
    train_keys = list(args.train_keys)
    log(f"Configuration: n_window={n_window}, seeds={seeds}, train_keys={train_keys}")

    log("Loading dataset and building matrices ...")
    ds = load_full_dataset()
    matrices = build_4xN_for_all_subsets(ds, n_window=n_window, stride=1)
    for k, (X, Y) in matrices.items():
        log(f"   {k:<22s}: X={X.shape}")

    cfg = CAECNNLocConfig(n_aps=4, n_window=n_window)

    all_rows = []
    histories_summary = []
    t_start = time.time()

    for train_key in train_keys:
        log(f"\n{'=' * 72}\nTraining subset = {train_key}\n{'=' * 72}")
        X_train, Y_train = matrices[train_key]
        others = {k: v[0] for k, v in matrices.items() if k != train_key}
        X_train_n, others_n, norm = normalize_train_test(X_train, others)
        log(f"   Normalisation: mean={norm['mean']:.3f}, std={norm['std']:.3f}")
        eval_subsets = {train_key: (X_train_n, Y_train)}
        eval_subsets.update({k: (others_n[k], matrices[k][1]) for k in others_n})

        for seed in seeds:
            t0 = time.time()
            regressor, _, history = train_one_seed(
                X_train_n, Y_train, seed, cfg, TRAIN_CFG,
                log_callback=lambda m: log("     " + m),
            )
            elapsed = time.time() - t0
            log(
                f"   seed={seed} | elapsed={elapsed:.1f}s | "
                f"pretrain val_loss min={min(history.pretrain_val_loss):.3f} | "
                f"finetune val_loss min={min(history.finetune_val_loss):.3f}"
            )
            df_seed = evaluate_on_subsets(
                regressor, eval_subsets, train_key,
                method_name=f"CAE-CNNLoc-2D-N{n_window}", seed=seed,
            )
            df_seed["n_window"] = n_window
            df_seed["pretrain_min_val_loss"] = float(min(history.pretrain_val_loss))
            df_seed["finetune_min_val_loss"] = float(min(history.finetune_val_loss))
            df_seed["train_elapsed_s"] = float(elapsed)
            all_rows.append(df_seed)
            histories_summary.append(
                {
                    "train_set": train_key,
                    "seed": seed,
                    "elapsed_s": elapsed,
                    "pretrain_min_val_loss": float(min(history.pretrain_val_loss)),
                    "finetune_min_val_loss": float(min(history.finetune_val_loss)),
                    "pretrain_epochs_done": len(history.pretrain_loss),
                    "finetune_epochs_done": len(history.finetune_loss),
                }
            )

            partial = pd.concat(all_rows, ignore_index=True)
            partial.to_csv(RESULTS / "02_cnnloc_full_results_partial.csv", index=False)

    total = time.time() - t_start
    log(f"\n=== Full run done in {total / 60:.1f} min ===")

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(RESULTS / "02_cnnloc_full_results.csv", index=False)
    log(f"Saved {len(df)} rows to results/02_cnnloc_full_results.csv")

    pd.DataFrame(histories_summary).to_csv(
        RESULTS / "02_cnnloc_full_history.csv", index=False
    )

    summary = aggregate_results(df, groupby=("train_set", "test_set"))
    summary = summary.round(3)
    summary.to_csv(RESULTS / "02_cnnloc_full_summary.csv", index=False)

    log("\nFinal summary (mean error in metres, aggregated across seeds):\n")
    pivot = summary.pivot(index="train_set", columns="test_set", values="mean_error")
    print(pivot.to_string())

    log("\nCross-device + cross-pose only:")
    cross = df[df["train_set"] != df["test_set"]]
    log(f"   mean error = {cross['error'].mean():.3f} m")
    log(f"   median     = {cross['error'].median():.3f} m")
    log(f"   p95        = {np.percentile(cross['error'], 95):.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
