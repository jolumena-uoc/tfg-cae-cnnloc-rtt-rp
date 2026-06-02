"""Genera las figuras adicionales del Anexo C de la PEC4.

- Mapa de calor del error mediano (m) de CAE-CNNLoc 2D-Temporal en la
  matriz train x test (a partir de results/02_cnnloc_full_summary.csv).
- Gráfica de barras del barrido del hiperparámetro N=8/16/32 entrenado en
  POCO#STANDING (a partir de results/tune_nwindow_summary.csv).

Ambas figuras se exportan a formato PDF en results/ y se copian también a
la carpeta Figuras/resultados/ del proyecto Overleaf.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OVERLEAF_FIGS = Path(r"C:/TFG/Overleaf/Figuras/resultados")


def heatmap_cnnloc(out_pdf: Path) -> None:
    df = pd.read_csv(RESULTS / "02_cnnloc_full_summary.csv")
    train_sets = ["POCO#STANDING", "POCO#TRIPOD", "SAMSUNG#STANDING", "SAMSUNG#TRIPOD"]
    matrix = (
        df.pivot(index="train_set", columns="test_set", values="median_error")
        .reindex(index=train_sets, columns=train_sets)
        .to_numpy()
    )

    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    im = ax.imshow(matrix, cmap="viridis", vmin=0, vmax=np.nanmax(matrix))

    short = ["POCO\nSTAND", "POCO\nTRIP", "SAMS\nSTAND", "SAMS\nTRIP"]
    ax.set_xticks(range(4), short)
    ax.set_yticks(range(4), short)
    ax.set_xlabel("Subconjunto de evaluación (test)")
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
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)


def barchart_n_sweep(out_pdf: Path) -> None:
    df = pd.read_csv(RESULTS / "tune_nwindow_summary.csv")
    test_order = ["POCO#STANDING", "POCO#TRIPOD", "SAMSUNG#STANDING", "SAMSUNG#TRIPOD"]
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
    ax.set_xlabel("Subconjunto de evaluación (test)")
    ax.set_ylabel("Error medio (m)")
    ax.set_title(
        "Barrido del tamaño de ventana N (entrenamiento en POCO#STANDING)",
    )
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(title="Tamaño N", loc="upper left")

    fig.tight_layout()
    fig.savefig(out_pdf)
    fig.savefig(out_pdf.with_suffix(".png"), dpi=200)
    plt.close(fig)


def main() -> None:
    heatmap_pdf = RESULTS / "anx_cnnloc_heatmap.pdf"
    barchart_pdf = RESULTS / "anx_n_sweep.pdf"

    heatmap_cnnloc(heatmap_pdf)
    barchart_n_sweep(barchart_pdf)

    OVERLEAF_FIGS.mkdir(parents=True, exist_ok=True)
    for src in (heatmap_pdf, barchart_pdf):
        shutil.copy2(src, OVERLEAF_FIGS / src.name)
        print(f"Copiado {src.name} -> {OVERLEAF_FIGS}")


if __name__ == "__main__":
    main()
