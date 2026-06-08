"""Agrega los CSVs por semilla de CAE-CNNLoc y por par (train, test) de los
baselines en los CSVs resumen que aparecen en la memoria del TFG.

Lee:
    results/01_baselines_centroid.csv
    results/01_baselines_knn_dtc_per_pair.csv
    results/02_cnnloc_<subset>_seed<N>_summary.csv  (cinco semillas por subset)
    o, en su defecto, results/02_cnnloc_summary.csv

Escribe:
    results/01_baselines_summary.csv
    results/01_baselines_per_sample.csv
    results/02_cnnloc_summary.csv  (consolidado por semilla)
    results/02_cnnloc_aggregated_summary.csv  (agg. sobre cinco semillas)
    results/tabla_global.csv
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
RES = REPO_ROOT / "results"

SUBSETS = ["POCO#STANDING", "POCO#TRIPOD", "SAMSUNG#STANDING", "SAMSUNG#TRIPOD"]


def scenario_label(train: str, test: str) -> str:
    if train == test:
        return "auto"
    t_dev, t_pose = train.split("#")
    e_dev, e_pose = test.split("#")
    if t_dev == e_dev and t_pose != e_pose:
        return "cross-pose"
    if t_dev != e_dev and t_pose == e_pose:
        return "cross-device"
    return "cross-both"


def stats_from_errors(errs):
    errs = np.asarray(errs, dtype=float)
    return dict(
        mean_error=float(errs.mean()),
        std_error=float(errs.std()),
        median_error=float(np.median(errs)),
        p75_error=float(np.percentile(errs, 75)),
        p95_error=float(np.percentile(errs, 95)),
        max_error=float(errs.max()),
        n_samples=int(len(errs)),
    )


def prepare_baselines() -> pd.DataFrame:
    """Genera summary y per_sample de los baselines (centroide y KNN-DtC k=1
    con ventana de 5 s)."""
    rows_summary = []
    rows_per_sample = []

    cent = pd.read_csv(RES / "01_baselines_centroid.csv")
    cent_5s = cent[cent["window"] == "5s"].copy()
    for ss in SUBSETS:
        sub = cent_5s[cent_5s["SET"] == ss]
        if len(sub) == 0:
            continue
        s = stats_from_errors(sub["error"])
        for scen_train in SUBSETS:
            rows_summary.append({
                "method": "centroid",
                "train_set": scen_train,
                "test_set": ss,
                "scenario": scenario_label(scen_train, ss),
                **s,
            })
        for _, r in sub.iterrows():
            for scen_train in SUBSETS:
                rows_per_sample.append({
                    "method": "centroid",
                    "train_set": scen_train,
                    "test_set": ss,
                    "scenario": scenario_label(scen_train, ss),
                    "error": float(r["error"]),
                })

    knn = pd.read_csv(RES / "01_baselines_knn_dtc_per_pair.csv")
    knn_k1_5s = knn[(knn["k"] == 1) & (knn["window"] == "5s")].copy()
    for tr in SUBSETS:
        for te in SUBSETS:
            sub = knn_k1_5s[(knn_k1_5s["train_set"] == tr) & (knn_k1_5s["test_set"] == te)]
            if len(sub) == 0:
                continue
            scen = scenario_label(tr, te)
            s = stats_from_errors(sub["error"])
            rows_summary.append({
                "method": "knn_dtc_k1",
                "train_set": tr,
                "test_set": te,
                "scenario": scen,
                **s,
            })
            for _, r in sub.iterrows():
                rows_per_sample.append({
                    "method": "knn_dtc_k1",
                    "train_set": tr,
                    "test_set": te,
                    "scenario": scen,
                    "error": float(r["error"]),
                })

    df_sum = pd.DataFrame(rows_summary)
    df_sum.to_csv(RES / "01_baselines_summary.csv", index=False)
    print(f"Escrito 01_baselines_summary.csv: {len(df_sum)} filas")

    df_ps = pd.DataFrame(rows_per_sample)
    df_ps.to_csv(RES / "01_baselines_per_sample.csv", index=False)
    print(f"Escrito 01_baselines_per_sample.csv: {len(df_ps)} filas")
    return df_sum


def prepare_cnnloc() -> pd.DataFrame | None:
    """Agrega CAE-CNNLoc sobre las semillas. Reconstruye desde los archivos
    individuales por seed si están disponibles."""
    src = RES / "02_cnnloc_summary.csv"
    seed_files = sorted(RES.glob("02_cnnloc_*_seed*_summary.csv"))
    if seed_files:
        df = pd.concat([pd.read_csv(p) for p in seed_files], ignore_index=True)
        df.to_csv(src, index=False)
        print(f"Reconstruido 02_cnnloc_summary.csv desde {len(seed_files)} ficheros por semilla")
    elif src.exists():
        df = pd.read_csv(src)
        print(f"Usando 02_cnnloc_summary.csv existente: {len(df)} filas")
    else:
        print("AVISO: no hay archivos por semilla ni consolidado para CAE-CNNLoc.")
        return None

    agg = df.groupby(["train_set", "test_set", "scenario"], as_index=False).agg(
        mean_error_mean=("mean_error", "mean"),
        mean_error_std=("mean_error", "std"),
        median_error_mean=("median_error", "mean"),
        p75_error_mean=("p75_error", "mean"),
        p95_error_mean=("p95_error", "mean"),
        max_error_mean=("max_error", "mean"),
        n_samples=("n_samples", "first"),
        n_seeds=("seed", "nunique"),
    )
    agg = agg.rename(columns={
        "mean_error_mean": "mean_error",
        "mean_error_std": "std_error",
        "median_error_mean": "median_error",
        "p75_error_mean": "p75_error",
        "p95_error_mean": "p95_error",
        "max_error_mean": "max_error",
    })
    agg.to_csv(RES / "02_cnnloc_aggregated_summary.csv", index=False)
    print(f"Escrito 02_cnnloc_aggregated_summary.csv: {len(agg)} filas, "
          f"n_seeds={int(agg['n_seeds'].max())}")
    return agg


def global_table(base: pd.DataFrame, cnn: pd.DataFrame | None) -> None:
    """Tabla resumen comparativa por escenario (sólo escenarios cruzados)."""
    rows = []
    for scen in ["cross-pose", "cross-device", "cross-both"]:
        c = base[(base["method"] == "centroid") & (base["scenario"] == scen)]
        c_dedup = c.drop_duplicates("test_set")
        m_c = c_dedup["mean_error"].mean()
        md_c = c_dedup["median_error"].mean()

        k = base[(base["method"] == "knn_dtc_k1") & (base["scenario"] == scen)]
        m_k = k["mean_error"].mean()
        md_k = k["median_error"].mean()

        if cnn is not None:
            n = cnn[cnn["scenario"] == scen]
            m_n = n["mean_error"].mean()
            s_n = n["std_error"].mean()
            md_n = n["median_error"].mean()
        else:
            m_n, s_n, md_n = (np.nan, np.nan, np.nan)
        rows.append(dict(
            scenario=scen,
            cent_mean=m_c, cent_median=md_c,
            knn_mean=m_k, knn_median=md_k,
            cnn_mean=m_n, cnn_std=s_n, cnn_median=md_n,
        ))
    df = pd.DataFrame(rows)
    df.to_csv(RES / "tabla_global.csv", index=False, float_format="%.3f")
    print("\nTabla global por escenario (cross-*):")
    print(df.to_string(index=False, float_format="%.3f"))
    print("\nEscrita en tabla_global.csv")


if __name__ == "__main__":
    print("=" * 80)
    print("AGREGADO DE RESULTADOS")
    print("=" * 80)
    base = prepare_baselines()
    cnn = prepare_cnnloc()
    global_table(base, cnn)
    print("\nListo.")
