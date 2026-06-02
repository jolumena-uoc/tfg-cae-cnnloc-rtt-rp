"""Keras implementation of CAE-CNNLoc 2D-Temporal for WiFi RTT.

The architecture follows the CAE-CNNLoc proposal of Kargar-Barzi et al.
(2024), adapted to a 4xN AP-time fingerprint matrix:

* Encoder: stack of Conv2D + BatchNorm + MaxPool blocks with progressively
  larger filter banks (16, 32, 64).
* Decoder (only used for unsupervised pre-training): symmetric stack of
  UpSampling2D + Conv2DTranspose blocks.
* Regression head (used in the supervised fine-tuning phase): Conv2D +
  BatchNorm + Flatten + Dropout + Dense -> linear output (x, y).

The core idea is to first train the autoencoder with a reconstruction
loss (MSE) and then drop the decoder, keep the encoder weights, attach
the regression head and fine-tune end-to-end with a coordinate-MSE loss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass(frozen=True)
class CAECNNLocConfig:
    """Hyper-parameters of CAE-CNNLoc 2D-Temporal."""

    n_aps: int = 4
    n_window: int = 16
    encoder_filters: Tuple[int, int, int] = (16, 32, 64)
    head_filters: int = 64
    dropout: float = 0.5
    learning_rate: float = 1e-3
    activation: str = "relu"


def _conv_block(x, filters: int, kernel: Tuple[int, int], pool: bool, name: str,
                activation: str = "relu") -> tf.Tensor:
    x = layers.Conv2D(filters, kernel, padding="same", name=f"{name}_conv")(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation(activation, name=f"{name}_act")(x)
    if pool:
        x = layers.MaxPool2D(pool_size=(1, 2), padding="same", name=f"{name}_pool")(x)
    return x


def build_encoder(cfg: CAECNNLocConfig) -> models.Model:
    """Encoder model that maps a 4xN matrix to a latent feature map."""
    inputs = layers.Input(shape=(cfg.n_aps, cfg.n_window, 1), name="encoder_input")
    x = inputs
    for i, f in enumerate(cfg.encoder_filters):
        x = _conv_block(
            x,
            filters=f,
            kernel=(3, 3),
            pool=True,
            name=f"enc_b{i+1}",
            activation=cfg.activation,
        )
    return models.Model(inputs, x, name="encoder")


def build_decoder(cfg: CAECNNLocConfig, latent_shape) -> models.Model:
    """Decoder model symmetric to the encoder for autoencoder pre-training."""
    inputs = layers.Input(shape=latent_shape[1:], name="decoder_input")
    x = inputs
    for i, f in enumerate(reversed(cfg.encoder_filters)):
        x = layers.UpSampling2D(size=(1, 2), name=f"dec_b{i+1}_up")(x)
        x = layers.Conv2DTranspose(
            f, kernel_size=(3, 3), padding="same", name=f"dec_b{i+1}_convt",
        )(x)
        x = layers.BatchNormalization(name=f"dec_b{i+1}_bn")(x)
        x = layers.Activation(cfg.activation, name=f"dec_b{i+1}_act")(x)
    # Final layer back to a single channel
    x = layers.Conv2D(1, (3, 3), padding="same", name="dec_out")(x)
    return models.Model(inputs, x, name="decoder")


def build_autoencoder(cfg: CAECNNLocConfig) -> Tuple[models.Model, models.Model, models.Model]:
    """Return (autoencoder, encoder, decoder) sharing encoder weights.

    The autoencoder output is cropped/resized to match the input's temporal
    dimension (necessary because of the rounding produced by max-pooling).
    """
    encoder = build_encoder(cfg)
    latent_shape = encoder.output_shape
    decoder = build_decoder(cfg, latent_shape)

    inputs = encoder.input
    latent = encoder(inputs)
    decoded = decoder(latent)

    # The decoder output may be longer than n_window by one or two
    # samples due to upsampling. Crop to the original temporal length.
    decoded = layers.Cropping2D(((0, 0), (0, max(0, decoded.shape[2] - cfg.n_window))),
                                name="dec_crop")(decoded)
    autoencoder = models.Model(inputs, decoded, name="cae_cnnloc_ae")
    return autoencoder, encoder, decoder


def build_regressor(cfg: CAECNNLocConfig, encoder: models.Model) -> models.Model:
    """Build the supervised regressor by attaching the head to the encoder."""
    inputs = layers.Input(shape=(cfg.n_aps, cfg.n_window, 1), name="reg_input")
    latent = encoder(inputs)
    x = layers.Conv2D(cfg.head_filters, kernel_size=(2, 2), padding="same",
                       name="head_conv")(latent)
    x = layers.BatchNormalization(name="head_bn")(x)
    x = layers.Activation(cfg.activation, name="head_act")(x)
    x = layers.Flatten(name="head_flatten")(x)
    x = layers.Dropout(cfg.dropout, name="head_dropout")(x)
    outputs = layers.Dense(2, activation="linear", name="coords")(x)
    return models.Model(inputs, outputs, name="cae_cnnloc_regressor")


def compile_autoencoder(model: models.Model, cfg: CAECNNLocConfig) -> models.Model:
    optimizer = tf.keras.optimizers.Nadam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def compile_regressor(model: models.Model, cfg: CAECNNLocConfig) -> models.Model:
    optimizer = tf.keras.optimizers.Nadam(learning_rate=cfg.learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model
