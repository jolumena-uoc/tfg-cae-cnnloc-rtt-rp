"""Regenera `results/02_cnnloc_curves.pdf` y `results/02_cnnloc_cdf.pdf`.

- `02_cnnloc_curves.pdf`: curvas de pérdida de un par (subset, semilla)
  representativo (POCO#STANDING semilla 0) para las dos fases del
  entrenamiento (pre-entrenamiento del CAE y ajuste fino del regresor).
- `02_cnnloc_cdf.pdf`: matriz 4x4 con las CDFs del error de CAE-CNNLoc
  2D-Temporal en cada par (entrenamiento, evaluación). La diagonal queda
  vacía y cada panel indica el error medio.

Lee `results/02_cnnloc_curves.csv` y `results/02_cnnloc_per_sample.csv`,
producidos por `scripts/run_cnnloc_full.py` y `scripts/aggregate_summary.py`.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS = REPO_ROOT / "results"

SUBSETS = ["POCO#STANDING", "POCO#TRIPOD", "SAMSUNG#STANDING", "SAMSUNG#TRIPOD"]
SCEN_COLOR = {
    "cross-pose": "#ff7f0e",
    "cross-device": "#2ca02c",
    "cross-both": "#d62728",
}


def cdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    return v, np.arange(1, len(v) + 1) / len(v)


def fig_curves() -> Path:
    src = RESULTS / "02_cnnloc_curves.csv"
    if not src.exists():
        raise SystemExit(
            f"No se encuentra {src}. Lanza primero "
            f"`python scripts/run_cnnloc_full.py`."
        )
    df = pd.read_csv(src)
    rep = df[(df["train_set"] == "POCO#STANDING") & (df["seed"] == 0)]
    if rep.empty:
        rep = df[df["seed"] == df["seed"].min()]

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 7.6))
    p1 = rep[rep["phase"] == "pretrain"]
    if len(p1):
        axes[0].plot(p1["epoch"], p1["loss"], lw=1.8, color="#1f77b4", label="train")
        axes[0].plot(p1["epoch"], p1["val_loss"], lw=1.8, color="#ff7f0e", label="val")
    axes[0].set_title("Fase 1: pre-entrenamiento del CAE", fontsize=10)
    axes[0].set_xlabel("\u00e9poca")
    axes[0].set_ylabel("MSE reconstrucci\u00f3n")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9)

    p2 = rep[rep["phase"] == "finetune"]
    if len(p2):
        axes[1].plot(p2["epoch"], p2["loss"], lw=1.8, color="#1f77b4", label="train")
        axes[1].plot(p2["epoch"], p2["val_loss"], lw=1.8, color="#ff7f0e", label="val")
    axes[1].set_title("Fase 2: ajuste fino del regresor", fontsize=10)
    axes[1].set_xlabel("\u00e9poca")
    axes[1].set_ylabel("MSE coords. (m$^2$)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9)

    fig.suptitle(
        "Curvas de p\u00e9rdida CAE-CNNLoc 2D-T (semilla representativa: POCO-STANDING_seed0)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RESULTS / "02_cnnloc_curves.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def fig_cdf_matrix() -> Path:
    src = RESULTS / "02_cnnloc_per_sample.csv"
    if not src.exists():
        raise SystemExit(
            f"No se encuentra {src}. Lanza primero "
            f"`python scripts/run_cnnloc_full.py`."
        )
    df = pd.read_csv(src)
    fig, axes = plt.subplots(4, 4, figsize=(13, 11), sharex=True, sharey=True)
    for i, tr in enumerate(SUBSETS):
        for j, te in enumerate(SUBSETS):
            ax = axes[i, j]
            if tr == te:
                ax.axis("off")
                if i == 0:
                    ax.set_title(te.replace("SAMSUNG", "S24U").replace("#", "\\#"),
                                 fontsize=9)
                continue
            sub = df[(df["train_set"] == tr) & (df["test_set"] == te)]
            if len(sub):
                v, c = cdf(sub["error"].values)
                col = SCEN_COLOR[sub["scenario"].iloc[0]]
                ax.plot(v, c, lw=1.8, color=col)
                ax.text(0.05, 0.93, f"media={sub['error'].mean():.2f} m",
                        transform=ax.transAxes, fontsize=8, color="black")
            ax.set_xlim(0, 6)
            ax.set_ylim(0, 1)
            ax.grid(True, alpha=0.3)
            if i == 0:
                ax.set_title(te.replace("SAMSUNG", "S24U").replace("#", "\\#"), fontsize=9)
            if j == 0:
                ax.set_ylabel(tr.replace("SAMSUNG", "S24U").replace("#", "\\#"), fontsize=9)
            if i == 3:
                ax.set_xlabel("Error (m)", fontsize=8)
    fig.suptitle(
        "CDFs CAE-CNNLoc 2D-T ($N=16$) por los 12 pares cruzados "
        "(entrenamiento, evaluaci\u00f3n).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = RESULTS / "02_cnnloc_cdf.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    out_curves = fig_curves()
    print(f"Generado: {out_curves}")
    out_cdf = fig_cdf_matrix()
    print(f"Generado: {out_cdf}")


if __name__ == "__main__":
    main()
