"""
src/feature_engineering.py
==========================
All feature engineering logic for the Airline Delay DNN project.

Usage
-----
    from src.feature_engineering import build_features
    df_features, le_carrier, le_airport = build_features(df_raw)
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


# ── Constants ──────────────────────────────────────────────────────────────
LEAKY_COLS = [
    'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct',
    'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay',
    'security_delay', 'late_aircraft_delay', 'arr_del15',
]

FEATURE_COLS = [
    'month_sin',          # Cyclic month — sine component
    'month_cos',          # Cyclic month — cosine component
    'year_norm',          # Normalised year [0, 1]
    'log_arr_flights',    # log1p(arr_flights) — reduces skewness
    'cancel_rate',        # arr_cancelled / arr_flights
    'divert_rate',        # arr_diverted / arr_flights
    'carrier_hist_delay', # Rolling 12-month carrier delay rate (shift=1)
    'airport_hist_delay', # Rolling 12-month airport delay rate (shift=1)
    'is_summer',          # Binary: June / July / August
    'is_holiday_month',   # Binary: November / December
    'carrier_enc',        # Label-encoded carrier (embedding index)
    'airport_enc',        # Label-encoded airport (embedding index)
]

TARGET_COL = 'delay_rate'

SUMMER_MONTHS  = [6, 7, 8]
HOLIDAY_MONTHS = [11, 12]
ROLLING_WINDOW = 12   # months of history for rolling features
MIN_PERIODS    = 3    # minimum months before rolling mean is computed


def build_features(
    df: pd.DataFrame,
    le_carrier: LabelEncoder = None,
    le_airport: LabelEncoder = None,
) -> tuple:
    """
    Full feature engineering pipeline.

    Takes the raw Airline_Delay_Cause DataFrame and returns a clean
    DataFrame with all 12 engineered features + target + identifiers.

    Parameters
    ----------
    df : pd.DataFrame
        Raw loaded CSV — all 21 original columns must be present.
    le_carrier : LabelEncoder or None
        Pre-fitted carrier encoder. If None, a new one is fit on this data.
        Pass a fitted encoder when transforming val/test to ensure consistency.
    le_airport : LabelEncoder or None
        Pre-fitted airport encoder. Same rule as le_carrier.

    Returns
    -------
    df_out : pd.DataFrame
        Feature-engineered DataFrame sorted by carrier/airport/year/month.
    le_carrier : LabelEncoder
        Fitted carrier LabelEncoder.
    le_airport : LabelEncoder
        Fitted airport LabelEncoder.

    Notes
    -----
    - Leaky delay-cause columns are dropped before any feature is created.
    - Rolling history uses shift(1) to prevent target leakage.
    - Year normalisation parameters are computed from this dataset —
      pass the same encoder for val/test to use training statistics.
    """
    df = df.copy()

    # ── Step 1: Drop rows without a computable target ──────────────
    df = df.dropna(subset=['arr_flights', 'arr_del15'])
    df = df[df['arr_flights'] > 0].reset_index(drop=True)

    # ── Step 2: Compute target ─────────────────────────────────────
    df[TARGET_COL] = df['arr_del15'] / df['arr_flights']

    # ── Step 3: Drop leaky columns ─────────────────────────────────
    df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns])

    # ── Step 4: Sort chronologically ──────────────────────────────
    df = df.sort_values(['carrier', 'airport', 'year', 'month']).reset_index(drop=True)

    # ── Step 5: Cyclic month encoding ─────────────────────────────
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # ── Step 6: Normalised year ────────────────────────────────────
    year_min = df['year'].min()
    year_max = df['year'].max()
    df['year_norm'] = (df['year'] - year_min) / (year_max - year_min)

    # ── Step 7: Log-transformed flight volume ──────────────────────
    df['log_arr_flights'] = np.log1p(df['arr_flights'])

    # ── Step 8: Cancellation & diversion rates ─────────────────────
    df['cancel_rate'] = df['arr_cancelled'] / df['arr_flights']
    df['divert_rate'] = df['arr_diverted'].fillna(0.0) / df['arr_flights']

    # ── Step 9: Rolling carrier delay history ──────────────────────
    # Sort by carrier + time, then compute trailing 12-month mean
    # shift(1) ensures current month is excluded — prevents leakage
    df = df.sort_values(['carrier', 'year', 'month']).reset_index(drop=True)
    df['carrier_hist_delay'] = (
        df.groupby('carrier')[TARGET_COL]
          .transform(lambda x: x.shift(1)
                                .rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS)
                                .mean())
    )

    # ── Step 10: Rolling airport delay history ─────────────────────
    df = df.sort_values(['airport', 'year', 'month']).reset_index(drop=True)
    df['airport_hist_delay'] = (
        df.groupby('airport')[TARGET_COL]
          .transform(lambda x: x.shift(1)
                                .rolling(ROLLING_WINDOW, min_periods=MIN_PERIODS)
                                .mean())
    )

    # ── Step 11: Binary seasonal flags ────────────────────────────
    df['is_summer']        = df['month'].isin(SUMMER_MONTHS).astype(int)
    df['is_holiday_month'] = df['month'].isin(HOLIDAY_MONTHS).astype(int)

    # ── Step 12: Label encoding ────────────────────────────────────
    if le_carrier is None:
        le_carrier = LabelEncoder()
        df['carrier_enc'] = le_carrier.fit_transform(df['carrier'])
    else:
        df['carrier_enc'] = le_carrier.transform(df['carrier'])

    if le_airport is None:
        le_airport = LabelEncoder()
        df['airport_enc'] = le_airport.fit_transform(df['airport'])
    else:
        df['airport_enc'] = le_airport.transform(df['airport'])

    # ── Final sort & return ────────────────────────────────────────
    df = df.sort_values(['carrier', 'airport', 'year', 'month']).reset_index(drop=True)

    return df, le_carrier, le_airport


def get_feature_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a summary table of all engineered features.

    Parameters
    ----------
    df : pd.DataFrame
        Output of build_features().

    Returns
    -------
    pd.DataFrame
        Columns: feature, null_count, null_pct, mean, std, min, max.
    """
    rows = []
    for col in FEATURE_COLS + [TARGET_COL]:
        if col not in df.columns:
            continue
        rows.append({
            'feature'   : col,
            'null_count': df[col].isna().sum(),
            'null_pct'  : round(df[col].isna().mean() * 100, 3),
            'mean'      : round(df[col].mean(), 4),
            'std'        : round(df[col].std(), 4),
            'min'        : round(df[col].min(), 4),
            'max'        : round(df[col].max(), 4),
        })
    return pd.DataFrame(rows)
