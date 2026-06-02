"""Regenera 02_cnnloc_curves.{pdf,png} con disposición 2x1 vertical.

Entrena UN único (train_key, semilla) con N=16 ---el mismo elegido en la
memoria--- para volver a obtener las curvas de pre-entrenamiento y de ajuste
fino, y guarda la figura. Se elige POCO#STANDING / seed=0 por consistencia con
el notebook 02_cnnloc.ipynb.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cnnloc_rtt.data import (  # noqa: E402
    build_4xN_for_all_subsets,
    load_full_dataset,
    normalize_train_test,
)
from cnnloc_rtt.models import CAECNNLocConfig  # noqa: E402
from cnnloc_rtt.train import TrainConfig, train_one_seed  # noqa: E402

RESULTS = ROOT / "results"
OVERLEAF_FIG = Path(r"C:/TFG/Overleaf/Figuras/resultados/cnnloc_curves.pdf")

TRAIN_KEY = "POCO#STANDING"
SEED = 0
N_WINDOW = 16
STRIDE = 1


def main() -> int:
    t0 = time.time()
    print(f"Loading dataset and building matrices N={N_WINDOW}...")
    ds = load_full_dataset()
    matrices = build_4xN_for_all_subsets(ds, n_window=N_WINDOW, stride=STRIDE)

    X_train, Y_train = matrices[TRAIN_KEY]
    others = {k: v[0] for k, v in matrices.items() if k != TRAIN_KEY}
    X_train_n, _, norm = normalize_train_test(X_train, others)
    print(
        f"  Normalisation mean={norm['mean']:.3f}, std={norm['std']:.3f} | "
        f"X_train_n={X_train_n.shape}"
    )

    cfg = CAECNNLocConfig(n_aps=4, n_window=N_WINDOW)
    train_cfg = TrainConfig(
        pretrain_epochs=60,
        finetune_epochs=80,
        batch_size=64,
        val_split=0.2,
        early_stopping_patience=12,
        verbose=0,
    )
    print(f"Training one seed ({TRAIN_KEY}, seed={SEED})...")
    _, _, history = train_one_seed(
        X_train_n,
        Y_train,
        SEED,
        cfg,
        train_cfg,
        log_callback=lambda m: print("  ", m),
    )

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.8))
    axes[0].plot(history.pretrain_loss, label="train")
    axes[0].plot(history.pretrain_val_loss, label="val")
    axes[0].set_title(
        f"Fase 1 - Pre-entrenamiento CAE ({TRAIN_KEY}, semilla={SEED})"
    )
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE de reconstrucción")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history.finetune_loss, label="train")
    axes[1].plot(history.finetune_val_loss, label="val")
    axes[1].set_title("Fase 2 - Ajuste fino del regresor")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel(r"MSE de coordenadas (m$^2$)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()

    png_out = RESULTS / "02_cnnloc_curves.png"
    pdf_out = RESULTS / "02_cnnloc_curves.pdf"
    fig.savefig(png_out, dpi=150, bbox_inches="tight")
    fig.savefig(pdf_out, bbox_inches="tight")
    plt.close(fig)

    OVERLEAF_FIG.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(pdf_out, OVERLEAF_FIG)
    print(f"\nGenerado: {pdf_out}")
    print(f"Copiado : {OVERLEAF_FIG}")
    print(f"Tiempo total: {(time.time() - t0)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
