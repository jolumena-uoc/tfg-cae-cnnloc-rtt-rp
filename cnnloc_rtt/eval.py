"""Evaluation utilities for CAE-CNNLoc 2D-Temporal.

Produces a long-format DataFrame compatible with the `error` column
schema used in the original reproducibility package, so that the three
methods (centroid, KNN-DtC, CAE-CNNLoc 2D-Temporal) can be combined
into a single dataframe and analysed jointly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd

from .utils import euclidean_error, summary_stats


def predict_errors(
    model,
    X: np.ndarray,
    Y: np.ndarray,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run the regressor on (X, Y) and return predictions and per-sample errors."""
    preds = model.predict(X, batch_size=batch_size, verbose=0)
    errors = euclidean_error(preds, Y)
    return preds, errors


def evaluate_on_subsets(
    model,
    subsets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    train_set_key: str,
    method_name: str = "CAE-CNNLoc-2D",
    seed: int | None = None,
) -> pd.DataFrame:
    """Evaluate the model across all subsets and return a long-format frame.

    Columns: ['method', 'train_set', 'test_set', 'seed', 'sample_idx',
    'pred_x', 'pred_y', 'true_x', 'true_y', 'error']
    """
    rows: List[dict] = []
    for k, (X, Y) in subsets.items():
        preds, errors = predict_errors(model, X, Y)
        for i in range(len(errors)):
            rows.append(
                {
                    "method": method_name,
                    "train_set": train_set_key,
                    "test_set": k,
                    "seed": seed if seed is not None else -1,
                    "sample_idx": i,
                    "pred_x": float(preds[i, 0]),
                    "pred_y": float(preds[i, 1]),
                    "true_x": float(Y[i, 0]),
                    "true_y": float(Y[i, 1]),
                    "error": float(errors[i]),
                }
            )
    return pd.DataFrame(rows)


def aggregate_results(
    df: pd.DataFrame,
    groupby: Iterable[str] = ("method", "train_set", "test_set"),
) -> pd.DataFrame:
    """Summary statistics (mean / std / median / p75 / p95 / max)."""
    agg = (
        df.groupby(list(groupby))["error"]
        .agg(
            mean_error="mean",
            std_error="std",
            median_error="median",
            p75_error=lambda s: float(np.percentile(s, 75)),
            p95_error=lambda s: float(np.percentile(s, 95)),
            max_error="max",
            n_samples="count",
        )
        .reset_index()
    )
    return agg
