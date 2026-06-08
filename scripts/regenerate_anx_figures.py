"""Regenera las figuras del anexo del documento:

- `results/anx_cnnloc_heatmap.pdf`: mapa de calor del error mediano (m) de
  CAE-CNNLoc 2D-Temporal ($N=16$) en la matriz train x test, agregado sobre
  las cinco semillas.
- `results/anx_n_sweep.pdf`: gráfica de barras del barrido del hiperparámetro
  $N \\in \\{8, 16, 32\\}$ entrenado sobre POCO#STANDING.

Lee `results/02_cnnloc_aggregated_summary.csv` (producido por
`scripts/aggregate_summary.py`) y `results/tune_nwindow_summary.csv`
(producido por `scripts/tune_nwindow.py`).
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


def heatmap_cnnloc(out_pdf: Path) -> None:
    src = RESULTS / "02_cnnloc_aggregated_summary.csv"
    if not src.exists():
        raise SystemExit(
            f"No se encuentra {src}. Lanza primero "
            f"`python scripts/aggregate_summary.py`."
        )
    df = pd.read_csv(src)
    matrix = (
        df.pivot(index="train_set", columns="test_set", values="median_error")
        .reindex(index=SUBSETS, columns=SUBSETS)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=np.nanmax(matrix))

    short = ["POCO\nSTAND", "POCO\nTRIP", "SAMS\nSTAND", "SAMS\nTRIP"]
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("Subconjunto de evaluaci\u00f3n (test)")
    ax.set_ylabel("Subconjunto de entrenamiento (train)")
    ax.set_title(
        "CAE-CNNLoc 2D-Temporal (N=16)\nError mediano (m) por par (train, test)",
    )

    for i in range(4):
        for j in range(4):
            v = matrix[i, j]
            color = "white" if v > 1.5 else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", color=color, fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Error mediano (m)")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def barchart_n_sweep(out_pdf: Path) -> None:
    src = RESULTS / "tune_nwindow_summary.csv"
    if not src.exists():
        raise SystemExit(
            f"No se encuentra {src}. Lanza primero "
            f"`python scripts/tune_nwindow.py`."
        )
    df = pd.read_csv(src)
    test_order = SUBSETS
    n_values = [8, 16, 32]

    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.25
    x = np.arange(len(test_order))
    colors = {8: "#5b8def", 16: "#2ca02c", 32: "#d62728"}

    for offset, n in enumerate(n_values):
        sub = df[df["n_window"] == n].set_index("test_set").reindex(test_order)
        ax.bar(
            x + (offset - 1) * width,
            sub["mean_error"],
            width=width,
            yerr=sub["std_error"],
            capsize=3,
            label=f"N = {n}",
            color=colors[n],
        )

    short = ["POCO\nSTAND", "POCO\nTRIP", "SAMS\nSTAND", "SAMS\nTRIP"]
    ax.set_xticks(x, short)
    ax.set_xlabel("Subconjunto de evaluaci\u00f3n (test)")
    ax.set_ylabel("Error medio (m)")
    ax.set_title(
        "Barrido del tama\u00f1o de ventana N (entrenamiento en POCO#STANDING)",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Tama\u00f1o N", loc="upper left")

    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    heatmap_pdf = RESULTS / "anx_cnnloc_heatmap.pdf"
    barchart_pdf = RESULTS / "anx_n_sweep.pdf"
    heatmap_cnnloc(heatmap_pdf)
    print(f"Generado: {heatmap_pdf}")
    barchart_n_sweep(barchart_pdf)
    print(f"Generado: {barchart_pdf}")


if __name__ == "__main__":
    main()
