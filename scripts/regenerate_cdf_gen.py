"""Regenera 03_comparativa_cdf_generalisation.{pdf,png} con disposición 3x1 vertical.

Lee los resultados unificados ya producidos por el notebook 03_comparativa.ipynb.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OVERLEAF_FIG = Path(r"C:/TFG/Overleaf/Figuras/resultados/comparativa_cdf_generalisation.pdf")

UNIFIED = RESULTS / "03_unified_results.csv"
df_all = pd.read_csv(UNIFIED)


def plot_cdf(ax, errors, **kwargs):
    e = np.sort(np.asarray(errors, dtype=float))
    if len(e) == 0:
        return
    cdf = np.arange(1, len(e) + 1) / len(e)
    ax.plot(e, cdf, **kwargs)


FIXED_VARIANTS = {
    "KNN-DtC": "KNN-DtC-k1_w5s",
    "CAE-CNNLoc-2D": "CAE-CNNLoc-2D-N16",
}
LABEL_MAP = {
    "KNN-DtC": "KNN-DtC (k=1, 5s)",
    "CAE-CNNLoc-2D": "CAE-CNNLoc 2D-T (N=16)",
}

fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.6), sharex=True)
for ax, scen in zip(axes, ["cross-pose", "cross-device", "cross-both"]):
    for fam, color in [("KNN-DtC", "tab:blue"), ("CAE-CNNLoc-2D", "tab:red")]:
        var = FIXED_VARIANTS[fam]
        sub = df_all[
            (df_all["family"] == fam)
            & (df_all["variant"] == var)
            & (df_all["scenario"] == scen)
        ]
        if sub.empty:
            continue
        plot_cdf(ax, sub["error"], label=LABEL_MAP[fam], color=color, linewidth=1.8)
    ax.set_xlim(0, 8)
    ax.set_ylabel("CDF")
    ax.set_title(f"Escenario: {scen}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="lower right")
axes[-1].set_xlabel("Error de localización (m)")
fig.suptitle(
    "CDF del error: KNN-DtC vs CAE-CNNLoc 2D-Temporal por escenario", y=0.995
)
fig.tight_layout()

png_out = RESULTS / "03_comparativa_cdf_generalisation.png"
pdf_out = RESULTS / "03_comparativa_cdf_generalisation.pdf"
fig.savefig(png_out, dpi=150, bbox_inches="tight")
fig.savefig(pdf_out, bbox_inches="tight")
plt.close(fig)

# Copia al directorio Overleaf para que LaTeX lo encuentre.
OVERLEAF_FIG.parent.mkdir(parents=True, exist_ok=True)
shutil.copyfile(pdf_out, OVERLEAF_FIG)
print(f"Generado: {pdf_out}")
print(f"Copiado : {OVERLEAF_FIG}")
