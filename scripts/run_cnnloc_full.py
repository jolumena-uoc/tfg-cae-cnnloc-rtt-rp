"""Orquesta la ejecución completa de CAE-CNNLoc 2D-Temporal: para cada uno
de los cuatro subconjuntos (POCO/SAMSUNG x STANDING/TRIPOD) y cada una de
las cinco semillas (0..4), invoca `run_cnnloc_one.py` y a continuación
consolida todos los CSVs por semilla en cuatro CSVs unificados:

    results/02_cnnloc_per_sample.csv
    results/02_cnnloc_summary.csv
    results/02_cnnloc_history.csv
    results/02_cnnloc_curves.csv

Estos CSVs son el insumo de `aggregate_summary.py`, que produce los resúmenes
agregados sobre semillas que aparecen en la memoria del TFG.

Uso:
    python scripts/run_cnnloc_full.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
ONE_SCRIPT = REPO_ROOT / "scripts" / "run_cnnloc_one.py"
OUT_DIR = REPO_ROOT / "results"
PYTHON = sys.executable

SUBSETS = ["POCO#STANDING", "POCO#TRIPOD", "SAMSUNG#STANDING", "SAMSUNG#TRIPOD"]
SEEDS = [0, 1, 2, 3, 4]
N_WINDOW = 16
STRIDE = 2
PRETRAIN_EPOCHS = 30
FINETUNE_EPOCHS = 40
BATCH_SIZE = 32


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["TF_CPP_MIN_LOG_LEVEL"] = "2"
    env["TF_ENABLE_ONEDNN_OPTS"] = "0"
    total = len(SUBSETS) * len(SEEDS)
    done = 0
    t_start = time.time()

    for subset in SUBSETS:
        for seed in SEEDS:
            done += 1
            log(f"=== [{done}/{total}] {subset} seed={seed} ===")
            t0 = time.time()
            cmd = [
                PYTHON, str(ONE_SCRIPT),
                "--train", subset,
                "--seed", str(seed),
                "--n-window", str(N_WINDOW),
                "--stride", str(STRIDE),
                "--pretrain-epochs", str(PRETRAIN_EPOCHS),
                "--finetune-epochs", str(FINETUNE_EPOCHS),
                "--batch-size", str(BATCH_SIZE),
                "--out-dir", str(OUT_DIR),
            ]
            try:
                ret = subprocess.run(
                    cmd, env=env, check=False,
                    stdout=sys.stdout, stderr=subprocess.STDOUT,
                )
                if ret.returncode != 0:
                    log(f"  !! FALLO subprocess (code={ret.returncode})")
            except Exception as e:
                log(f"  !! EXC subprocess: {e}")
            log(f"  elapsed={time.time() - t0:.1f}s "
                f"(total {(time.time() - t_start) / 60:.1f} min)")

    log("Consolidando CSVs ...")
    per_sample, summary, history, curves = [], [], [], []
    for subset in SUBSETS:
        for seed in SEEDS:
            tag = f"{subset.replace('#', '-')}_seed{seed}"
            ps = OUT_DIR / f"02_cnnloc_{tag}.csv"
            sm = OUT_DIR / f"02_cnnloc_{tag}_summary.csv"
            hi = OUT_DIR / f"02_cnnloc_{tag}_history.csv"
            cv = OUT_DIR / f"02_cnnloc_{tag}_curves.csv"
            if ps.exists():
                per_sample.append(pd.read_csv(ps))
            if sm.exists():
                summary.append(pd.read_csv(sm))
            if hi.exists():
                history.append(pd.read_csv(hi))
            if cv.exists():
                curves.append(pd.read_csv(cv))
    if per_sample:
        pd.concat(per_sample, ignore_index=True).to_csv(
            OUT_DIR / "02_cnnloc_per_sample.csv", index=False)
    if summary:
        df_sum = pd.concat(summary, ignore_index=True)
        df_sum.to_csv(OUT_DIR / "02_cnnloc_summary.csv", index=False)
        pivot = df_sum.groupby("scenario")["mean_error"].agg(
            ["mean", "std", "count"]).round(3)
        log("Resumen final por escenario (mean error en m, agg semillas y subsets):")
        print(pivot.to_string())
    if history:
        pd.concat(history, ignore_index=True).to_csv(
            OUT_DIR / "02_cnnloc_history.csv", index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(
            OUT_DIR / "02_cnnloc_curves.csv", index=False)
    log(f"Total: {(time.time() - t_start) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
