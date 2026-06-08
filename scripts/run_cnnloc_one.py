"""Entrena UNA combinación (subconjunto, semilla) de CAE-CNNLoc 2D-Temporal
y guarda los resultados en `results/`.

Sigue el protocolo de Matey-Sanz et al. (2025): se entrena con TODAS las
muestras del subconjunto de entrenamiento y se evalúa con TODAS las muestras
de cada uno de los cuatro subconjuntos (POCO/SAMSUNG x STANDING/TRIPOD).
La diagonal (mismo subconjunto en train y test) se conserva en los CSVs por
trazabilidad pero no se utiliza en el análisis principal de la memoria.

Uso:
    python scripts/run_cnnloc_one.py --train POCO#STANDING --seed 0

Salidas (con `tag = <train>-<seed>`):
    results/02_cnnloc_<tag>.csv             # per-sample
    results/02_cnnloc_<tag>_summary.csv     # resumen por par (train, test)
    results/02_cnnloc_<tag>_history.csv     # tiempos y validación
    results/02_cnnloc_<tag>_curves.csv      # curvas (loss, val_loss) por época
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Paquete reproducible original de Matey-Sanz y Torres-Sospedra (proporciona
# `lib.model.Device` y `lib.model.MeasuringType`). Por defecto se busca como
# carpeta hermana del repo; se puede sobreescribir con la variable de entorno
# `MATEY_RTT_DIR`.
MATEY_DIR = Path(
    os.environ.get(
        "MATEY_RTT_DIR",
        REPO_ROOT.parent / "Comparative_Analysis_Android_RTT_main",
    )
)
if not MATEY_DIR.exists():
    raise SystemExit(
        f"No se encuentra el paquete reproducible de Matey-Sanz et al. en\n"
        f"  {MATEY_DIR}\n"
        f"Defínelo con la variable de entorno MATEY_RTT_DIR o colócalo como\n"
        f"carpeta hermana del repositorio."
    )
sys.path.insert(0, str(MATEY_DIR))

import numpy as np
import pandas as pd

from cnnloc_rtt.data import (
    build_4xN_matrices_for_subset,
    load_full_dataset,
    normalize_train_test,
)
from cnnloc_rtt.models import CAECNNLocConfig
from cnnloc_rtt.train import TrainConfig, train_one_seed
from lib.model import Device, MeasuringType


def scenario_label(train_key: str, test_key: str) -> str:
    if train_key == test_key:
        return "auto"
    tr_dev, tr_pose = train_key.split("#")
    te_dev, te_pose = test_key.split("#")
    if tr_dev == te_dev and tr_pose != te_pose:
        return "cross-pose"
    if tr_dev != te_dev and tr_pose == te_pose:
        return "cross-device"
    return "cross-both"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--train", required=True,
                    help="Subconjunto de entrenamiento, p.ej. POCO#STANDING")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--n-window", type=int, default=16)
    ap.add_argument("--stride", type=int, default=2,
                    help="Stride para reducir muestras (1 = denso, 2 = mitad)")
    ap.add_argument("--pretrain-epochs", type=int, default=30)
    ap.add_argument("--finetune-epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "results"),
                    help="Directorio de salida (por defecto <repo>/results)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = lambda m: print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

    log(f"Cargando dataset (train={args.train}, seed={args.seed}, "
        f"N={args.n_window}, stride={args.stride})...")
    ds = load_full_dataset()
    matrices = {}
    for dev_name in ("POCO", "SAMSUNG"):
        for mt_name in ("STANDING", "TRIPOD"):
            dev = getattr(Device, dev_name)
            mt = getattr(MeasuringType, mt_name)
            X, Y = build_4xN_matrices_for_subset(
                ds, dev, mt, n_window=args.n_window, stride=args.stride
            )
            matrices[f"{dev_name}#{mt_name}"] = (X, Y)
    del ds
    gc.collect()

    X_tr_raw, Y_tr = matrices[args.train]
    log(f"  Train completo: X_tr={X_tr_raw.shape}")

    eval_test = {k: (X, Y) for k, (X, Y) in matrices.items()}

    X_tr_n, others_n, norm = normalize_train_test(
        X_tr_raw, {k: v[0] for k, v in eval_test.items()}
    )
    eval_norm = {k: (others_n[k], eval_test[k][1]) for k in eval_test}
    log(f"  X_tr norm shape={X_tr_n.shape} mean={norm['mean']:.3f} "
        f"std={norm['std']:.3f}")

    cfg = CAECNNLocConfig(n_aps=4, n_window=args.n_window)
    train_cfg = TrainConfig(
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        val_split=0.2,
        early_stopping_patience=8,
        verbose=0,
    )

    t0 = time.time()
    log("  Entrenando CAE-CNNLoc 2D-Temporal ...")
    regressor, _, history = train_one_seed(
        X_tr_n, Y_tr, args.seed, cfg, train_cfg,
        log_callback=lambda m: log("    " + m),
    )
    elapsed = time.time() - t0
    log(f"  Entrenamiento terminado en {elapsed:.1f}s")

    rows, summary_rows = [], []
    for test_key, (X_te, Y_te) in eval_norm.items():
        preds = regressor.predict(X_te, batch_size=128, verbose=0)
        err = np.linalg.norm(preds - Y_te, axis=1)
        scenario = scenario_label(args.train, test_key)
        for i in range(len(err)):
            rows.append({
                "method": "CAE-CNNLoc-2D-N16",
                "train_set": args.train,
                "test_set": test_key,
                "scenario": scenario,
                "seed": args.seed,
                "sample_idx": i,
                "true_x": float(Y_te[i, 0]),
                "true_y": float(Y_te[i, 1]),
                "pred_x": float(preds[i, 0]),
                "pred_y": float(preds[i, 1]),
                "error": float(err[i]),
            })
        summary_rows.append({
            "method": "CAE-CNNLoc-2D-N16",
            "train_set": args.train,
            "test_set": test_key,
            "scenario": scenario,
            "seed": args.seed,
            "mean_error": float(err.mean()),
            "median_error": float(np.median(err)),
            "p75_error": float(np.percentile(err, 75)),
            "p95_error": float(np.percentile(err, 95)),
            "max_error": float(err.max()),
            "n_samples": int(len(err)),
        })

    tag = f"{args.train.replace('#', '-')}_seed{args.seed}"
    pd.DataFrame(rows).to_csv(
        out_dir / f"02_cnnloc_{tag}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        out_dir / f"02_cnnloc_{tag}_summary.csv", index=False)
    pd.DataFrame([{
        "train_set": args.train, "seed": args.seed,
        "elapsed_s": elapsed,
        "pretrain_epochs": len(history.pretrain_loss),
        "pretrain_min_val_loss": float(min(history.pretrain_val_loss)),
        "finetune_epochs": len(history.finetune_loss),
        "finetune_min_val_loss": float(min(history.finetune_val_loss)),
    }]).to_csv(out_dir / f"02_cnnloc_{tag}_history.csv", index=False)

    n_p = len(history.pretrain_loss)
    n_f = len(history.finetune_loss)
    curves_rows = []
    for ep in range(n_p):
        curves_rows.append({
            "train_set": args.train, "seed": args.seed, "phase": "pretrain",
            "epoch": ep,
            "loss": float(history.pretrain_loss[ep]),
            "val_loss": float(history.pretrain_val_loss[ep]),
        })
    for ep in range(n_f):
        curves_rows.append({
            "train_set": args.train, "seed": args.seed, "phase": "finetune",
            "epoch": ep,
            "loss": float(history.finetune_loss[ep]),
            "val_loss": float(history.finetune_val_loss[ep]),
        })
    pd.DataFrame(curves_rows).to_csv(
        out_dir / f"02_cnnloc_{tag}_curves.csv", index=False)

    log(f"  Resumen para {args.train} seed={args.seed}:")
    for sr in summary_rows:
        log(f"    {sr['scenario']:<13s} {sr['test_set']:<22s} "
            f"mean={sr['mean_error']:.3f} m  median={sr['median_error']:.3f} m")
    log("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
