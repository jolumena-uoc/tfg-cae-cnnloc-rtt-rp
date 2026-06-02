"""Smoke test for the CAE-CNNLoc 2D-Temporal architecture (Keras).

Run as:
    python tests/test_model.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Reduce TF noise
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

from cnnloc_rtt.data import build_4xN_for_all_subsets, load_full_dataset
from cnnloc_rtt.models import (
    CAECNNLocConfig,
    build_autoencoder,
    build_regressor,
    compile_autoencoder,
    compile_regressor,
)
from cnnloc_rtt.utils import set_global_seed


def main() -> int:
    set_global_seed(0)

    print("[1/5] Loading dataset and building matrices (N=16) ...")
    ds = load_full_dataset()
    matrices = build_4xN_for_all_subsets(ds, n_window=16, stride=1)
    train_key = "POCO#STANDING"
    X, Y = matrices[train_key]
    print(f"     train subset {train_key}: X={X.shape}, Y={Y.shape}")

    print("[2/5] Building CAE-CNNLoc 2D-Temporal models ...")
    cfg = CAECNNLocConfig(n_aps=4, n_window=16)
    ae, encoder, _ = build_autoencoder(cfg)
    reg = build_regressor(cfg, encoder)
    print(f"     AE   params: {ae.count_params()}")
    print(f"     Reg  params: {reg.count_params()}")

    print("[3/5] Compiling and running a 1-epoch CAE pre-training ...")
    ae = compile_autoencoder(ae, cfg)
    sub = X[:256]
    history_ae = ae.fit(sub, sub, epochs=1, batch_size=64, verbose=0)
    print(f"     CAE 1ep loss = {history_ae.history['loss'][0]:.4f}")

    print("[4/5] Compiling and running a 1-epoch regressor fine-tuning ...")
    reg = compile_regressor(reg, cfg)
    history_reg = reg.fit(X[:256], Y[:256], epochs=1, batch_size=64, verbose=0)
    print(f"     Reg 1ep loss = {history_reg.history['loss'][0]:.4f}")

    print("[5/5] Inference shape check ...")
    preds = reg.predict(X[:8], verbose=0)
    assert preds.shape == (8, 2), f"unexpected pred shape {preds.shape}"
    print(f"     preds shape: {preds.shape}, sample: ({preds[0,0]:.2f}, {preds[0,1]:.2f})")

    print("\nAll model smoke tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
