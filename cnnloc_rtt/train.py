"""Two-phase training routine for CAE-CNNLoc 2D-Temporal.

Phase 1 - Unsupervised: train the autoencoder on the training subset using a
mean-squared-error reconstruction loss on the 4xN fingerprint matrices.

Phase 2 - Supervised: discard the decoder, attach the regression head to
the encoder and fine-tune end-to-end with a coordinate-MSE loss against
the (x, y) targets.

The routine supports multi-seed runs (mean +- std reporting) and
optional early stopping with validation loss patience.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .models import (
    CAECNNLocConfig,
    build_autoencoder,
    build_regressor,
    compile_autoencoder,
    compile_regressor,
)
from .utils import set_global_seed


@dataclass
class TrainConfig:
    """Hyper-parameters of the two-phase training procedure."""

    pretrain_epochs: int = 60
    finetune_epochs: int = 80
    batch_size: int = 64
    val_split: float = 0.2
    early_stopping_patience: int = 12
    verbose: int = 0  # passed to keras.fit


@dataclass
class TrainHistory:
    """Container with the per-phase loss curves of a single seed."""

    pretrain_loss: List[float] = field(default_factory=list)
    pretrain_val_loss: List[float] = field(default_factory=list)
    finetune_loss: List[float] = field(default_factory=list)
    finetune_val_loss: List[float] = field(default_factory=list)


def _stratified_split(
    Y: np.ndarray, val_split: float, rng: np.random.Generator
) -> Tuple[np.ndarray, np.ndarray]:
    """Stratified train/val split by reference point.

    `Y` is the (N, 2) target coordinates, used to derive the reference
    point identifier as a tuple.
    """
    keys = np.array([f"{x:.4f}|{y:.4f}" for x, y in Y])
    train_idx: List[int] = []
    val_idx: List[int] = []
    unique = np.unique(keys)
    for k in unique:
        members = np.where(keys == k)[0]
        rng.shuffle(members)
        n_val = max(1, int(round(len(members) * val_split)))
        val_idx.extend(members[:n_val].tolist())
        train_idx.extend(members[n_val:].tolist())
    return np.asarray(train_idx), np.asarray(val_idx)


def train_one_seed(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    seed: int,
    cfg: CAECNNLocConfig,
    train_cfg: TrainConfig,
    log_callback: Optional[Callable[[str], None]] = None,
) -> Tuple["object", "object", TrainHistory]:
    """Run the two phases for a given seed and return models and history."""
    import tensorflow as tf

    set_global_seed(seed)

    rng = np.random.default_rng(seed)
    train_idx, val_idx = _stratified_split(Y_train, train_cfg.val_split, rng)
    X_tr, X_va = X_train[train_idx], X_train[val_idx]
    Y_tr, Y_va = Y_train[train_idx], Y_train[val_idx]

    autoencoder, encoder, _ = build_autoencoder(cfg)
    autoencoder = compile_autoencoder(autoencoder, cfg)

    if log_callback:
        log_callback(
            f"[seed={seed}] Phase 1 (CAE pre-training): "
            f"X_tr={X_tr.shape}, X_va={X_va.shape}"
        )

    es_pre = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=train_cfg.early_stopping_patience,
        restore_best_weights=True,
    )
    history_pre = autoencoder.fit(
        X_tr,
        X_tr,
        validation_data=(X_va, X_va),
        epochs=train_cfg.pretrain_epochs,
        batch_size=train_cfg.batch_size,
        verbose=train_cfg.verbose,
        callbacks=[es_pre],
        shuffle=True,
    )

    regressor = build_regressor(cfg, encoder)
    regressor = compile_regressor(regressor, cfg)

    if log_callback:
        log_callback(
            f"[seed={seed}] Phase 2 (regressor fine-tuning)"
        )

    es_ft = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=train_cfg.early_stopping_patience,
        restore_best_weights=True,
    )
    history_ft = regressor.fit(
        X_tr,
        Y_tr,
        validation_data=(X_va, Y_va),
        epochs=train_cfg.finetune_epochs,
        batch_size=train_cfg.batch_size,
        verbose=train_cfg.verbose,
        callbacks=[es_ft],
        shuffle=True,
    )

    history = TrainHistory(
        pretrain_loss=list(history_pre.history.get("loss", [])),
        pretrain_val_loss=list(history_pre.history.get("val_loss", [])),
        finetune_loss=list(history_ft.history.get("loss", [])),
        finetune_val_loss=list(history_ft.history.get("val_loss", [])),
    )
    return regressor, encoder, history


def train_multiseed(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    seeds,
    cfg: CAECNNLocConfig,
    train_cfg: TrainConfig,
    log_callback: Optional[Callable[[str], None]] = None,
) -> List[Tuple["object", "object", TrainHistory]]:
    """Run train_one_seed for each seed; returns the list of results.

    Note: each call holds a Keras model in memory; if memory is tight,
    persist the regressors to disk inside the loop instead of returning
    them.
    """
    results = []
    for s in seeds:
        results.append(train_one_seed(X_train, Y_train, s, cfg, train_cfg, log_callback))
    return results
