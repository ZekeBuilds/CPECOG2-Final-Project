"""Keras models and the custom APDropout layer.

Inputs are scaled RSSI from `data.to_model_input`, where 0 means "AP not detected".
"""
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import keras  # noqa: E402
from keras import layers, ops  # noqa: E402

N_CLASSES = 13


@keras.saving.register_keras_serializable(package="group6")
class APDropout(layers.Layer):
    """Training-time AP outage.

    For each scan, draw a failure rate p ~ U(0, max_rate), then switch off each detected AP
    with probability p by setting it to 0 (the "not detected" value). Unlike standard dropout,
    survivors are NOT rescaled: a failed AP must look exactly like an absent one, and the
    remaining readings keep their true signal strength (proposal Assumption 1).
    Sampling p per scan exposes the network to light and heavy outages in the same batch.
    Identity at inference.
    """

    def __init__(self, max_rate=0.5, seed=None, **kwargs):
        super().__init__(**kwargs)
        if not 0.0 <= max_rate < 1.0:
            raise ValueError("max_rate must be in [0, 1)")
        self.max_rate = float(max_rate)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

    def call(self, x, training=None):
        if not training or self.max_rate == 0.0:
            return x
        p = keras.random.uniform((ops.shape(x)[0], 1), 0.0, self.max_rate, seed=self.seed_generator)
        keep = keras.random.uniform(ops.shape(x), seed=self.seed_generator) >= p
        return ops.where(keep, x, ops.zeros_like(x))

    def get_config(self):
        return {**super().get_config(), "max_rate": self.max_rate, "seed": self.seed}


def build_mlp(n_inputs=520, hidden=(512, 256), dropout=0.3, l2=1e-4, ap_dropout=0.0,
              lr=1e-3, seed=0, name=None):
    """MLP classifier. ap_dropout > 0 gives the robust variant; 0 gives the normal one."""
    reg = keras.regularizers.L2(l2) if l2 else None
    inputs = keras.Input((n_inputs,), name="rssi")
    # Seed offset keeps APDropout's mask stream independent of the Dropout layers' streams.
    x = APDropout(ap_dropout, seed=seed + 1000, name="ap_dropout")(inputs) if ap_dropout > 0 else inputs
    for i, units in enumerate(hidden):
        x = layers.Dense(units, kernel_regularizer=reg, name=f"dense_{i}")(x)
        x = layers.BatchNormalization(name=f"bn_{i}")(x)
        x = layers.Activation("relu", name=f"relu_{i}")(x)
        x = layers.Dropout(dropout, seed=seed + i, name=f"dropout_{i}")(x)
    outputs = layers.Dense(N_CLASSES, activation="softmax", name="class")(x)
    model = keras.Model(inputs, outputs, name=name or ("mlp_apdrop" if ap_dropout > 0 else "mlp"))
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _compile_classifier(model, lr):
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def _head(x, units, dropout, reg, seed, prefix="head"):
    x = layers.Dropout(dropout, seed=seed + 50, name=f"{prefix}_in_dropout")(x)
    x = layers.Dense(units, kernel_regularizer=reg, name=f"{prefix}_dense")(x)
    x = layers.BatchNormalization(name=f"{prefix}_bn")(x)
    x = layers.Activation("relu", name=f"{prefix}_relu")(x)
    x = layers.Dropout(dropout, seed=seed + 51, name=f"{prefix}_dropout")(x)
    return layers.Dense(N_CLASSES, activation="softmax", name="class")(x)


def build_autoencoder(n_inputs=520, encoder=(256, 128), ap_dropout=0.0, lr=1e-3, seed=0):
    """Stacked autoencoder after Nowicki & Wietrzykowski / Kim et al.

    With ap_dropout > 0 it becomes a denoising autoencoder (Vincent et al. 2008): the input
    passes through APDropout but the reconstruction target is the clean scan.
    Returns (autoencoder, encoder_model); both share the encoder layers.
    """
    inputs = keras.Input((n_inputs,), name="rssi")
    x = APDropout(ap_dropout, seed=seed + 1000, name="ap_dropout")(inputs) if ap_dropout > 0 else inputs
    for i, units in enumerate(encoder):
        x = layers.Dense(units, activation="relu", name=f"enc_{i}")(x)
    code = x
    for i, units in enumerate(reversed(encoder[:-1])):
        x = layers.Dense(units, activation="relu", name=f"dec_{i}")(x)
    recon = layers.Dense(n_inputs, activation="sigmoid", name="recon")(x)
    ae = keras.Model(inputs, recon, name="autoencoder")
    ae.compile(optimizer=keras.optimizers.Adam(lr), loss="mse")
    return ae, keras.Model(inputs, code, name="encoder")


def build_ae_classifier(encoder_model, head_units=128, dropout=0.2, l2=1e-4, lr=1e-3, seed=0, name=None):
    """Pretrained encoder + classification head; the whole stack is fine-tuned."""
    reg = keras.regularizers.L2(l2) if l2 else None
    out = _head(encoder_model.output, head_units, dropout, reg, seed)
    ap = any(isinstance(l, APDropout) for l in encoder_model.layers)
    model = keras.Model(encoder_model.input, out, name=name or ("ae_mlp_apdrop" if ap else "ae_mlp"))
    return _compile_classifier(model, lr)


def build_cnn(n_inputs=520, proj=128, filters=(32, 64), kernel=5, dropout=0.3, l2=1e-4,
              ap_dropout=0.0, lr=1e-3, seed=0, name=None):
    """1D-CNN. AP columns have no spatial order, so a learned dense projection first maps the
    520 APs to `proj` features that the convolutions then scan, similar in spirit to
    Song et al.'s CNNLoc (SAE features -> 1D convolutions), but trained end to end."""
    reg = keras.regularizers.L2(l2) if l2 else None
    inputs = keras.Input((n_inputs,), name="rssi")
    x = APDropout(ap_dropout, seed=seed + 1000, name="ap_dropout")(inputs) if ap_dropout > 0 else inputs
    x = layers.Dense(proj, kernel_regularizer=reg, name="proj")(x)
    x = layers.BatchNormalization(name="proj_bn")(x)
    x = layers.Activation("relu", name="proj_relu")(x)
    x = layers.Reshape((proj, 1), name="to_sequence")(x)
    for i, f in enumerate(filters):
        x = layers.Conv1D(f, kernel, padding="same", kernel_regularizer=reg, name=f"conv_{i}")(x)
        x = layers.BatchNormalization(name=f"conv_bn_{i}")(x)
        x = layers.Activation("relu", name=f"conv_relu_{i}")(x)
        x = layers.MaxPooling1D(2, name=f"pool_{i}")(x)
    x = layers.Flatten(name="flatten")(x)
    out = _head(x, 128, dropout, reg, seed)
    model = keras.Model(inputs, out, name=name or ("cnn_apdrop" if ap_dropout > 0 else "cnn"))
    return _compile_classifier(model, lr)


def fit(model, x_train, y_train, x_val, y_val, epochs=150, batch_size=256, patience=15, verbose=0):
    """Early stopping on val loss, best weights restored. Returns the Keras History."""
    stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    return model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=epochs,
                     batch_size=batch_size, callbacks=[stop], verbose=verbose)


def predict_labels(model, x):
    return ops.convert_to_numpy(ops.argmax(model.predict(x, batch_size=2048, verbose=0), axis=1))
