"""Regenera `results/03_comparativa_cdf_generalisation.pdf`: CDFs comparativas
del centroide ponderado, KNN-DtC ($k=1$, ventana 5 s) y CAE-CNNLoc 2D-T
($N=16$) en los tres escenarios cruzados (cross-pose, cross-device, cross-both).

Lee `results/01_baselines_per_sample.csv` y `results/02_cnnloc_per_sample.csv`,
producidos por `scripts/aggregate_summary.py` y `scripts/run_cnnloc_full.py`.
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

METHOD_COLOR = {
    "Centroide": "#1f77b4",
    "KNN-DtC ($k{=}1$)": "#2ca02c",
    "CAE-CNNLoc 2D-T ($N{=}16$)": "#d62728",
}


def cdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    return v, np.arange(1, len(v) + 1) / len(v)


def main() -> None:
    base_src = RESULTS / "01_baselines_per_sample.csv"
    cnn_src = RESULTS / "02_cnnloc_per_sample.csv"
    if not base_src.exists() or not cnn_src.exists():
        raise SystemExit(
            "No se encuentran los CSVs de muestras. Lanza antes:\n"
            "  python scripts/aggregate_summary.py\n"
            "  python scripts/run_cnnloc_full.py"
        )
    base = pd.read_csv(base_src)
    cnn = pd.read_csv(cnn_src)

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 9.6), sharex=True)
    for ax, scen in zip(axes, ["cross-pose", "cross-device", "cross-both"]):
        sub_c = base[(base["method"] == "centroid") & (base["scenario"] == scen)]
        if len(sub_c):
            v, c = cdf(sub_c["error"].values)
            ax.plot(v, c, lw=2, color=METHOD_COLOR["Centroide"], label="Centroide")
        sub_k = base[(base["method"] == "knn_dtc_k1") & (base["scenario"] == scen)]
        if len(sub_k):
            v, c = cdf(sub_k["error"].values)
            ax.plot(v, c, lw=2, color=METHOD_COLOR["KNN-DtC ($k{=}1$)"],
                    label="KNN-DtC ($k{=}1$)")
        sub_n = cnn[cnn["scenario"] == scen]
        if len(sub_n):
            v, c = cdf(sub_n["error"].values)
            ax.plot(v, c, lw=2, color=METHOD_COLOR["CAE-CNNLoc 2D-T ($N{=}16$)"],
                    label="CAE-CNNLoc 2D-T ($N{=}16$)")
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("CDF")
        ax.set_title(f"Escenario {scen}", fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    axes[-1].set_xlabel("Error (m)")
    fig.tight_layout()

    out = RESULTS / "03_comparativa_cdf_generalisation.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generado: {out}")


if __name__ == "__main__":
    main()
