"""Regenera `results/01_baselines_cdf.pdf`: CDFs del centroide ponderado y
KNN-DtC ($k=1$, ventana 5 s) para los tres escenarios cruzados (cross-pose,
cross-device, cross-both).

Lee `results/01_baselines_per_sample.csv` (producido por
`scripts/aggregate_summary.py`) y escribe el PDF correspondiente.
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

SCEN_COLOR = {
    "cross-pose": "#ff7f0e",
    "cross-device": "#2ca02c",
    "cross-both": "#d62728",
}


def cdf(values):
    v = np.sort(np.asarray(values, dtype=float))
    return v, np.arange(1, len(v) + 1) / len(v)


def main() -> None:
    src = RESULTS / "01_baselines_per_sample.csv"
    if not src.exists():
        raise SystemExit(
            f"No se encuentra {src}. Lanza primero "
            f"`python scripts/aggregate_summary.py`."
        )
    df = pd.read_csv(src)

    fig, axes = plt.subplots(2, 1, figsize=(7.0, 8.0), sharex=True)
    titles = ["Centroide ponderado", "KNN-DtC ($k{=}1$)"]
    methods = ["centroid", "knn_dtc_k1"]
    cross_scenarios = ["cross-pose", "cross-device", "cross-both"]
    for ax, t, m in zip(axes, titles, methods):
        ax.set_title(t, fontsize=11)
        for scen in cross_scenarios:
            sub = df[(df["method"] == m) & (df["scenario"] == scen)]
            if len(sub) == 0:
                continue
            v, c = cdf(sub["error"].values)
            ax.plot(v, c, lw=2, color=SCEN_COLOR[scen], label=scen)
        ax.set_xlim(0, 6)
        ax.set_ylim(0, 1)
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="lower right")
    axes[-1].set_xlabel("Error (m)")
    fig.tight_layout()

    out = RESULTS / "01_baselines_cdf.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Generado: {out}")


if __name__ == "__main__":
    main()
