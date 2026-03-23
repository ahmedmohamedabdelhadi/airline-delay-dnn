"""
src/model.py
============
DNN architecture definition and training utilities.

Usage
-----
    from src.model import build_dnn, make_callbacks, train_model

    model   = build_dnn(n_carriers=30, n_airports=419)
    history = train_model(model, train_inputs, y_train, val_inputs, y_val)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks, regularizers


def build_dnn(
    n_cont        : int   = 10,
    n_carriers    : int   = 30,
    n_airports    : int   = 419,
    emb_dim_c     : int   = 8,
    emb_dim_a     : int   = 16,
    hidden_units  : tuple = (256, 128, 64),
    dropout_rate  : float = 0.3,
    l2_reg        : float = 1e-4,
    learning_rate : float = 1e-3,
) -> Model:
    """
    Build and compile the Airline Delay Deep Neural Network.

    Architecture
    ------------
    3 inputs → embeddings → concatenate → N×(Dense+BN+ReLU+Dropout) → sigmoid

    The model has three separate input branches:
    1. Continuous features (10 floats) — passed through dense layers directly
    2. Carrier index (int) — looked up in an Embedding(n_carriers, emb_dim_c)
    3. Airport index (int) — looked up in an Embedding(n_airports, emb_dim_a)

    All branches are concatenated and passed through the hidden layers.
    The output uses sigmoid to guarantee predictions in [0, 1].

    Parameters
    ----------
    n_cont        : Number of continuous/binary input features
    n_carriers    : Carrier vocabulary size (number of unique carriers)
    n_airports    : Airport vocabulary size (number of unique airports)
    emb_dim_c     : Carrier embedding dimension
    emb_dim_a     : Airport embedding dimension
    hidden_units  : Tuple of hidden layer widths
    dropout_rate  : Dropout fraction (applied after each hidden layer)
    l2_reg        : L2 weight decay for Dense and Embedding layers
    learning_rate : Adam initial learning rate

    Returns
    -------
    keras.Model — compiled, ready for .fit()
    """
    reg = regularizers.l2(l2_reg)

    # ── Input layers ───────────────────────────────────────────────
    inp_cont    = keras.Input(shape=(n_cont,), name='continuous', dtype='float32')
    inp_carrier = keras.Input(shape=(1,),      name='carrier',    dtype='int32')
    inp_airport = keras.Input(shape=(1,),      name='airport',    dtype='int32')

    # ── Embedding branches ──────────────────────────────────────────
    # Each categorical integer is mapped to a dense learned vector.
    # Rule of thumb for embedding dim: min(50, vocab_size // 2)
    emb_c = layers.Embedding(
        n_carriers, emb_dim_c,
        embeddings_regularizer=reg,
        name='emb_carrier'
    )(inp_carrier)
    emb_c = layers.Flatten(name='flat_carrier')(emb_c)   # (batch, emb_dim_c)

    emb_a = layers.Embedding(
        n_airports, emb_dim_a,
        embeddings_regularizer=reg,
        name='emb_airport'
    )(inp_airport)
    emb_a = layers.Flatten(name='flat_airport')(emb_a)   # (batch, emb_dim_a)

    # ── Merge all branches ──────────────────────────────────────────
    # Total width = n_cont + emb_dim_c + emb_dim_a
    x = layers.Concatenate(name='concat')([inp_cont, emb_c, emb_a])

    # ── Hidden layers: Dense → BatchNorm → ReLU → Dropout ──────────
    # use_bias=False because BatchNorm has its own bias (beta parameter)
    for i, units in enumerate(hidden_units):
        x = layers.Dense(
            units,
            use_bias=False,
            kernel_regularizer=reg,
            name=f'dense_{i}'
        )(x)
        x = layers.BatchNormalization(name=f'bn_{i}')(x)
        x = layers.Activation('relu', name=f'relu_{i}')(x)
        x = layers.Dropout(dropout_rate, name=f'drop_{i}')(x)

    # ── Output: sigmoid ensures prediction ∈ [0, 1] ────────────────
    output = layers.Dense(1, activation='sigmoid', name='output')(x)

    # ── Compile ────────────────────────────────────────────────────
    model = Model(
        inputs  = [inp_cont, inp_carrier, inp_airport],
        outputs = output,
        name    = 'AirlineDelayDNN'
    )
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate),
        loss      = 'mae',
        metrics   = ['mae', keras.metrics.RootMeanSquaredError(name='rmse')]
    )
    return model


def make_callbacks(
    run_name   : str = 'dnn',
    ckpt_path  : str = '../models/checkpoints/',
    patience_es: int = 15,
    patience_lr: int = 7,
) -> list:
    """
    Create the standard training callback suite.

    Callbacks
    ---------
    1. EarlyStopping      — stops training if val_mae doesn't improve for
                            `patience_es` epochs; restores best weights.
    2. ReduceLROnPlateau  — halves the learning rate if val_mae plateaus
                            for `patience_lr` epochs; floor at 1e-6.
    3. ModelCheckpoint    — saves the full model whenever val_mae improves.
    4. TensorBoard        — writes logs to ckpt_path/logs/run_name/.

    Parameters
    ----------
    run_name    : str — used to name checkpoint files and log directories
    ckpt_path   : str — directory for checkpoint and log files
    patience_es : int — EarlyStopping patience (epochs)
    patience_lr : int — ReduceLROnPlateau patience (epochs)

    Returns
    -------
    list of keras.callbacks.Callback
    """
    return [
        callbacks.EarlyStopping(
            monitor='val_mae', patience=patience_es,
            restore_best_weights=True, verbose=1, mode='min'
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_mae', factor=0.5, patience=patience_lr,
            min_lr=1e-6, verbose=1, mode='min'
        ),
        callbacks.ModelCheckpoint(
            filepath=ckpt_path + f'{run_name}_best.keras',
            monitor='val_mae', save_best_only=True,
            save_weights_only=False, verbose=0, mode='min'
        ),
        callbacks.TensorBoard(
            log_dir=ckpt_path + f'logs/{run_name}',
            histogram_freq=0, write_graph=False
        ),
    ]


def pack_inputs(X_cont, X_carrier, X_airport) -> dict:
    """
    Pack split arrays into the dict format Keras named inputs expect.

    Parameters
    ----------
    X_cont    : float32 array (n, 10)
    X_carrier : int32   array (n,)
    X_airport : int32   array (n,)

    Returns
    -------
    dict with keys 'continuous', 'carrier', 'airport'
    """
    return {
        'continuous': X_cont,
        'carrier'   : X_carrier,
        'airport'   : X_airport,
    }


def train_model(
    model,
    train_inputs : dict,
    y_train      : np.ndarray,
    val_inputs   : dict,
    y_val        : np.ndarray,
    epochs       : int = 200,
    batch_size   : int = 512,
    cb_list      : list = None,
) -> keras.callbacks.History:
    """
    Train the DNN with the standard configuration.

    Parameters
    ----------
    model        : compiled keras.Model from build_dnn()
    train_inputs : dict — output of pack_inputs() for training data
    y_train      : np.ndarray — training targets
    val_inputs   : dict — output of pack_inputs() for validation data
    y_val        : np.ndarray — validation targets
    epochs       : int — maximum training epochs (early stopping will kick in)
    batch_size   : int — mini-batch size
    cb_list      : list of callbacks, or None (uses make_callbacks() defaults)

    Returns
    -------
    keras.callbacks.History
    """
    if cb_list is None:
        cb_list = make_callbacks()

    history = model.fit(
        train_inputs, y_train,
        validation_data = (val_inputs, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        callbacks       = cb_list,
        verbose         = 1,
    )
    return history
