"""Regenera 01_baselines_cdf.{pdf,png} con disposición 3x1 vertical.

Lee las CSVs ya producidas por el notebook 01_baselines.ipynb.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OVERLEAF_FIG = Path(r"C:/TFG/Overleaf/Figuras/resultados/baselines_cdf.pdf")

df_centroid = pd.read_csv(RESULTS / "01_baselines_centroid.csv")
df_knn = pd.read_csv(RESULTS / "01_baselines_knn_dtc.csv")


def plot_cdf(ax, errors, label, **kw):
    e = np.sort(np.asarray(errors, dtype=float))
    if len(e) == 0:
        return
    cdf = np.arange(1, len(e) + 1) / len(e)
    ax.plot(e, cdf, label=label, **kw)


fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.6), sharex=True)
for ax, w in zip(axes, ["raw", "1s", "5s"]):
    sub_c = df_centroid[df_centroid["window"] == w]
    plot_cdf(ax, sub_c["error"], "Centroide", color="tab:orange", linewidth=2)
    for k, color in zip(
        [1, 3, 5, 10],
        ["tab:blue", "tab:green", "tab:red", "tab:purple"],
    ):
        sub_k = df_knn[
            (df_knn["window"] == w) & (df_knn["method"] == f"KNN-DtC-k{k}")
        ]
        plot_cdf(ax, sub_k["error"], f"KNN-DtC k={k}", color=color, linewidth=1.5)
    ax.set_xlim(0, 8)
    ax.set_ylabel("CDF")
    ax.set_title(f"Ventana = {w}")
    ax.grid(alpha=0.3)
axes[-1].set_xlabel("Error de localización (m)")
axes[0].legend(fontsize=9, loc="lower right")
fig.suptitle("CDF del error: centroide ponderado vs KNN-DtC", y=0.995)
fig.tight_layout()

png_out = RESULTS / "01_baselines_cdf.png"
pdf_out = RESULTS / "01_baselines_cdf.pdf"
fig.savefig(png_out, dpi=150, bbox_inches="tight")
fig.savefig(pdf_out, bbox_inches="tight")
plt.close(fig)

OVERLEAF_FIG.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(pdf_out, OVERLEAF_FIG)
print(f"Generado: {pdf_out}")
print(f"Copiado : {OVERLEAF_FIG}")
