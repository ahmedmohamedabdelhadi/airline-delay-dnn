"""
src/data_loader.py
==================
Loads and validates the raw Airline Delay Cause CSV.

Usage
-----
    from src.data_loader import load_raw_data
    df = load_raw_data('data/raw/Airline_Delay_Cause.csv')
"""

import pandas as pd


# ── Expected schema ────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    'year', 'month', 'carrier', 'carrier_name', 'airport', 'airport_name',
    'arr_flights', 'arr_del15', 'carrier_ct', 'weather_ct', 'nas_ct',
    'security_ct', 'late_aircraft_ct', 'arr_cancelled', 'arr_diverted',
    'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay',
    'security_delay', 'late_aircraft_delay',
]

LEAKY_COLS = [
    'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct', 'late_aircraft_ct',
    'arr_delay', 'carrier_delay', 'weather_delay', 'nas_delay',
    'security_delay', 'late_aircraft_delay', 'arr_del15',
]


def load_raw_data(path: str, verbose: bool = True) -> pd.DataFrame:
    """
    Load the raw Airline Delay Cause CSV and run schema validation.

    Parameters
    ----------
    path : str
        Path to Airline_Delay_Cause.csv.
    verbose : bool
        Print a summary after loading.

    Returns
    -------
    pd.DataFrame
        Raw DataFrame — all 21 original columns present, no rows dropped.

    Raises
    ------
    AssertionError
        If any expected column is missing from the file.
    """
    df = pd.read_csv(path)

    # ── Schema validation ──────────────────────────────────────────────
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    assert len(missing) == 0, f"Missing columns: {missing}"

    if verbose:
        print(f"✅ Loaded: {path}")
        print(f"   Shape       : {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   Year range  : {df['year'].min()} → {df['year'].max()}")
        print(f"   Carriers    : {df['carrier'].nunique()}")
        print(f"   Airports    : {df['airport'].nunique()}")
        print(f"   Total nulls : {df.isnull().sum().sum():,}")

    return df


def get_data_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a per-column quality report for the raw DataFrame.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Columns: dtype, null_count, null_pct, nunique, is_leaky.
    """
    report = pd.DataFrame({
        'dtype'      : df.dtypes,
        'null_count' : df.isnull().sum(),
        'null_pct'   : (df.isnull().mean() * 100).round(2),
        'nunique'    : df.nunique(),
        'is_leaky'   : [c in LEAKY_COLS for c in df.columns],
    })
    return report
