"""
src/preprocessing.py
====================
Temporal split, imputation, and scaling for the Airline Delay DNN project.

Usage
-----
    from src.preprocessing import make_splits, preprocess

    train_df, val_df, test_df = make_splits(df_features)
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputer = preprocess(
        train_df, val_df, test_df
    )
"""

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.feature_engineering import FEATURE_COLS, TARGET_COL


# ── Split configuration ────────────────────────────────────────────────────
TRAIN_END  = 2018   # train on years ≤ this
VAL_START  = 2019   # validation starts here
VAL_END    = 2020   # validation ends here
TEST_START = 2021   # test starts here

# Continuous features that need StandardScaler
# Binary flags and integer encodings are deliberately excluded
CONTINUOUS_COLS = [
    'month_sin', 'month_cos', 'year_norm', 'log_arr_flights',
    'cancel_rate', 'divert_rate', 'carrier_hist_delay', 'airport_hist_delay',
]
CONT_IDX = [FEATURE_COLS.index(c) for c in CONTINUOUS_COLS]


def make_splits(df: pd.DataFrame) -> tuple:
    """
    Apply temporal train / validation / test split.

    Split strategy
    --------------
    - Train : year ≤ 2018  (2003–2018, ~78% of data)
    - Val   : 2019–2020    (~12% of data — includes COVID disruption)
    - Test  : year ≥ 2021  (~9% of data — post-COVID recovery)

    Why temporal and not random?
    A random split would allow future rows to appear in training,
    contaminating the rolling history features with future information.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features() — must contain a 'year' column.

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
    """
    train_df = df[df['year'] <= TRAIN_END].copy()
    val_df   = df[(df['year'] >= VAL_START) & (df['year'] <= VAL_END)].copy()
    test_df  = df[df['year'] >= TEST_START].copy()

    total = len(df)
    print(f"Train : {len(train_df):>7,} rows  "
          f"({train_df['year'].min()}–{train_df['year'].max()})  "
          f"{len(train_df)/total*100:.1f}%")
    print(f"Val   : {len(val_df):>7,} rows  "
          f"({val_df['year'].min()}–{val_df['year'].max()})  "
          f"{len(val_df)/total*100:.1f}%")
    print(f"Test  : {len(test_df):>7,} rows  "
          f"({test_df['year'].min()}–{test_df['year'].max()})  "
          f"{len(test_df)/total*100:.1f}%")

    return train_df, val_df, test_df


def preprocess(
    train_df   : pd.DataFrame,
    val_df     : pd.DataFrame,
    test_df    : pd.DataFrame,
    imputer    : SimpleImputer = None,
    scaler     : StandardScaler = None,
) -> tuple:
    """
    Impute missing values and scale continuous features.

    Golden rule: fit on train, transform on everything.
    All statistics (median, mean, std) come exclusively from train_df.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        Outputs of make_splits().
    imputer : SimpleImputer or None
        Pre-fitted imputer. If None, a new one is fit on train_df.
        Pass a fitted imputer when processing new data at inference time.
    scaler : StandardScaler or None
        Pre-fitted scaler. Same rule as imputer.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray  — shape (n, 12), dtype float32
    y_train, y_val, y_test : np.ndarray  — shape (n,),    dtype float32
    scaler  : fitted StandardScaler
    imputer : fitted SimpleImputer
    """
    # ── Extract raw feature matrices ────────────────────────────────
    X_train_raw = train_df[FEATURE_COLS].values.astype('float32')
    X_val_raw   = val_df[FEATURE_COLS].values.astype('float32')
    X_test_raw  = test_df[FEATURE_COLS].values.astype('float32')

    # ── Imputation (median from train only) ─────────────────────────
    if imputer is None:
        imputer = SimpleImputer(strategy='median')
        imputer.fit(X_train_raw)

    X_train_imp = imputer.transform(X_train_raw)
    X_val_imp   = imputer.transform(X_val_raw)
    X_test_imp  = imputer.transform(X_test_raw)

    # ── Scaling (StandardScaler on continuous cols from train only) ──
    if scaler is None:
        scaler = StandardScaler()
        scaler.fit(X_train_imp[:, CONT_IDX])

    X_train = X_train_imp.copy()
    X_val   = X_val_imp.copy()
    X_test  = X_test_imp.copy()

    X_train[:, CONT_IDX] = scaler.transform(X_train_imp[:, CONT_IDX])
    X_val[:,   CONT_IDX] = scaler.transform(X_val_imp[:,  CONT_IDX])
    X_test[:,  CONT_IDX] = scaler.transform(X_test_imp[:, CONT_IDX])

    # ── Target vectors ──────────────────────────────────────────────
    y_train = train_df[TARGET_COL].values.astype('float32')
    y_val   = val_df[TARGET_COL].values.astype('float32')
    y_test  = test_df[TARGET_COL].values.astype('float32')

    # ── Verify: no nulls remain ──────────────────────────────────────
    assert np.isnan(X_train).sum() == 0, "Nulls remain in X_train"
    assert np.isnan(X_val).sum()   == 0, "Nulls remain in X_val"
    assert np.isnan(X_test).sum()  == 0, "Nulls remain in X_test"

    print(f"✅ Preprocessing complete")
    print(f"   X_train : {X_train.shape}  nulls={np.isnan(X_train).sum()}")
    print(f"   X_val   : {X_val.shape}    nulls={np.isnan(X_val).sum()}")
    print(f"   X_test  : {X_test.shape}   nulls={np.isnan(X_test).sum()}")

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, imputer


def split_inputs(X: np.ndarray) -> tuple:
    """
    Split a (n, 12) feature matrix into the three DNN input streams.

    Parameters
    ----------
    X : np.ndarray  shape (n, 12)

    Returns
    -------
    X_cont    : float32 (n, 10) — continuous + binary features
    X_carrier : int32   (n,)    — carrier label index
    X_airport : int32   (n,)    — airport label index
    """
    X_cont    = X[:, :10].astype('float32')
    X_carrier = X[:, 10].astype('int32')
    X_airport = X[:, 11].astype('int32')
    return X_cont, X_carrier, X_airport
