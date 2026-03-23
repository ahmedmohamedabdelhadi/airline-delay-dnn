"""
tests/test_features.py
======================
Unit tests for feature engineering and preprocessing logic.

Run with:
    pytest tests/test_features.py -v

These tests verify that:
- Feature values stay within expected ranges
- No data leakage exists in rolling features
- Cyclic encoding has correct mathematical properties
- Preprocessing obeys the fit-on-train rule
- Target computation is correct
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.feature_engineering import (
    build_features, get_feature_summary,
    FEATURE_COLS, TARGET_COL, SUMMER_MONTHS, HOLIDAY_MONTHS,
)
from src.preprocessing import make_splits, preprocess, split_inputs


# ══════════════════════════════════════════════════════════════════════
# Fixtures — minimal synthetic dataset
# ══════════════════════════════════════════════════════════════════════

@pytest.fixture
def raw_df():
    """
    Minimal synthetic DataFrame that mimics the real Airline_Delay_Cause CSV.
    Contains 3 carriers × 2 airports × 3 years × 12 months = 216 rows.
    """
    np.random.seed(42)
    rows = []
    for year in [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022]:
        for month in range(1, 13):
            for carrier in ['AA', 'DL', 'UA']:
                for airport in ['JFK', 'LAX']:
                    arr_flights   = np.random.randint(50, 500)
                    arr_del15     = np.random.randint(0, arr_flights // 2)
                    arr_cancelled = np.random.randint(0, 10)
                    arr_diverted  = np.random.randint(0, 5)
                    rows.append({
                        'year'               : year,
                        'month'              : month,
                        'carrier'            : carrier,
                        'carrier_name'       : f'{carrier} Airlines',
                        'airport'            : airport,
                        'airport_name'       : f'{airport} Airport',
                        'arr_flights'        : float(arr_flights),
                        'arr_del15'          : float(arr_del15),
                        'carrier_ct'         : float(np.random.randint(0, 10)),
                        'weather_ct'         : float(np.random.randint(0, 5)),
                        'nas_ct'             : float(np.random.randint(0, 8)),
                        'security_ct'        : 0.0,
                        'late_aircraft_ct'   : float(np.random.randint(0, 10)),
                        'arr_cancelled'      : float(arr_cancelled),
                        'arr_diverted'       : float(arr_diverted),
                        'arr_delay'          : float(np.random.randint(100, 5000)),
                        'carrier_delay'      : float(np.random.randint(50, 2000)),
                        'weather_delay'      : float(np.random.randint(0, 500)),
                        'nas_delay'          : float(np.random.randint(0, 1000)),
                        'security_delay'     : 0.0,
                        'late_aircraft_delay': float(np.random.randint(0, 1000)),
                    })
    return pd.DataFrame(rows)


@pytest.fixture
def features_df(raw_df):
    """Feature-engineered DataFrame from the synthetic raw data."""
    df, _, _ = build_features(raw_df)
    return df


# ══════════════════════════════════════════════════════════════════════
# 1. Target variable tests
# ══════════════════════════════════════════════════════════════════════

class TestTargetVariable:

    def test_delay_rate_range(self, features_df):
        """delay_rate must always be in [0, 1]."""
        assert features_df[TARGET_COL].between(0, 1).all(), \
            "delay_rate contains values outside [0, 1]"

    def test_delay_rate_formula(self, raw_df, features_df):
        """delay_rate = arr_del15 / arr_flights for every row."""
        merged = features_df.merge(
            raw_df[['year', 'month', 'carrier', 'airport',
                    'arr_del15', 'arr_flights']],
            on=['year', 'month', 'carrier', 'airport'],
            how='left'
        )
        expected = merged['arr_del15'] / merged['arr_flights']
        np.testing.assert_allclose(
            features_df[TARGET_COL].values,
            expected.values,
            rtol=1e-5,
            err_msg="delay_rate does not equal arr_del15 / arr_flights"
        )

    def test_no_null_target(self, features_df):
        """Target column must have no nulls after feature engineering."""
        assert features_df[TARGET_COL].isna().sum() == 0, \
            "delay_rate contains null values"


# ══════════════════════════════════════════════════════════════════════
# 2. Cyclic encoding tests
# ══════════════════════════════════════════════════════════════════════

class TestCyclicEncoding:

    def test_month_sin_range(self, features_df):
        """month_sin must be in [-1, 1]."""
        assert features_df['month_sin'].between(-1, 1).all()

    def test_month_cos_range(self, features_df):
        """month_cos must be in [-1, 1]."""
        assert features_df['month_cos'].between(-1, 1).all()

    def test_unit_circle_identity(self, features_df):
        """sin²(x) + cos²(x) = 1 for every row (Pythagorean identity)."""
        identity = features_df['month_sin']**2 + features_df['month_cos']**2
        np.testing.assert_allclose(
            identity.values, 1.0, atol=1e-6,
            err_msg="Cyclic encoding violates sin²+cos²=1"
        )

    def test_january_december_proximity(self, features_df):
        """
        January and December should be close in cyclic space.
        Euclidean distance between their (sin, cos) pairs must be < 0.6.
        (They are 1 month apart on the cycle, not 11.)
        """
        jan = features_df[features_df['month'] == 1][['month_sin','month_cos']].iloc[0]
        dec = features_df[features_df['month'] == 12][['month_sin','month_cos']].iloc[0]
        dist = np.sqrt((jan['month_sin'] - dec['month_sin'])**2 +
                       (jan['month_cos'] - dec['month_cos'])**2)
        assert dist < 0.6, \
            f"January and December are too far apart in cyclic space: dist={dist:.4f}"

    def test_june_july_are_far_from_december(self, features_df):
        """
        June and December should be far apart in cyclic space —
        they are 6 months apart (opposite sides of the cycle).
        """
        jun = features_df[features_df['month'] == 6][['month_sin','month_cos']].iloc[0]
        dec = features_df[features_df['month'] == 12][['month_sin','month_cos']].iloc[0]
        dist = np.sqrt((jun['month_sin'] - dec['month_sin'])**2 +
                       (jun['month_cos'] - dec['month_cos'])**2)
        assert dist > 1.5, \
            f"June and December should be far apart in cyclic space: dist={dist:.4f}"


# ══════════════════════════════════════════════════════════════════════
# 3. Ratio feature tests
# ══════════════════════════════════════════════════════════════════════

class TestRatioFeatures:

    def test_cancel_rate_range(self, features_df):
        """cancel_rate must be in [0, 1]."""
        assert features_df['cancel_rate'].between(0, 1).all(), \
            "cancel_rate contains values outside [0, 1]"

    def test_divert_rate_range(self, features_df):
        """divert_rate must be in [0, 1]."""
        assert features_df['divert_rate'].between(0, 1).all(), \
            "divert_rate contains values outside [0, 1]"

    def test_no_null_cancel_rate(self, features_df):
        """cancel_rate must have no nulls (arr_flights > 0 guaranteed)."""
        assert features_df['cancel_rate'].isna().sum() == 0

    def test_log_arr_flights_positive(self, features_df):
        """log1p(arr_flights) must always be > 0 (arr_flights >= 1)."""
        assert (features_df['log_arr_flights'] > 0).all()


# ══════════════════════════════════════════════════════════════════════
# 4. Binary flag tests
# ══════════════════════════════════════════════════════════════════════

class TestBinaryFlags:

    def test_is_summer_binary(self, features_df):
        """is_summer must only contain 0 or 1."""
        assert set(features_df['is_summer'].unique()).issubset({0, 1})

    def test_is_holiday_binary(self, features_df):
        """is_holiday_month must only contain 0 or 1."""
        assert set(features_df['is_holiday_month'].unique()).issubset({0, 1})

    def test_is_summer_correct_months(self, features_df):
        """is_summer must be 1 exactly for June, July, August."""
        for month in SUMMER_MONTHS:
            subset = features_df[features_df['month'] == month]
            assert (subset['is_summer'] == 1).all(), \
                f"is_summer should be 1 for month {month}"
        for month in set(range(1, 13)) - set(SUMMER_MONTHS):
            subset = features_df[features_df['month'] == month]
            assert (subset['is_summer'] == 0).all(), \
                f"is_summer should be 0 for month {month}"

    def test_is_holiday_correct_months(self, features_df):
        """is_holiday_month must be 1 exactly for November, December."""
        for month in HOLIDAY_MONTHS:
            subset = features_df[features_df['month'] == month]
            assert (subset['is_holiday_month'] == 1).all(), \
                f"is_holiday_month should be 1 for month {month}"


# ══════════════════════════════════════════════════════════════════════
# 5. Leakage prevention tests
# ══════════════════════════════════════════════════════════════════════

class TestNoLeakage:

    def test_leaky_columns_absent(self, features_df):
        """
        Delay-cause columns must NOT appear in the feature-engineered DataFrame.
        These are post-hoc measurements that would not be available at prediction time.
        """
        leaky = [
            'carrier_ct', 'weather_ct', 'nas_ct', 'security_ct',
            'late_aircraft_ct', 'arr_delay', 'carrier_delay',
            'weather_delay', 'nas_delay', 'security_delay',
            'late_aircraft_delay', 'arr_del15',
        ]
        present = [c for c in leaky if c in features_df.columns]
        assert len(present) == 0, \
            f"Leaky columns found in feature DataFrame: {present}"

    def test_rolling_history_uses_past_only(self, features_df):
        """
        carrier_hist_delay at time T must only use data from T-1 and earlier.
        We verify this by checking that the first record for each carrier
        (which has no history) is NaN.
        """
        # The very first month of each carrier's history should be NaN
        # (shift=1 means there's no data before the first record)
        first_records = (
            features_df
            .sort_values(['carrier', 'year', 'month'])
            .groupby('carrier')
            .first()
            .reset_index()
        )
        # First records should have NaN carrier_hist_delay
        # (not enough history yet — min_periods=3 means first 3 are NaN)
        assert first_records['carrier_hist_delay'].isna().all(), \
            "carrier_hist_delay should be NaN for the first record of each carrier"

    def test_year_norm_range(self, features_df):
        """year_norm must be in [0, 1]."""
        assert features_df['year_norm'].between(0, 1).all(), \
            "year_norm contains values outside [0, 1]"


# ══════════════════════════════════════════════════════════════════════
# 6. Preprocessing tests
# ══════════════════════════════════════════════════════════════════════

class TestPreprocessing:

    def test_temporal_split_no_overlap(self, features_df):
        """Train, val, and test splits must not share any rows."""
        train_df, val_df, test_df = make_splits(features_df)

        train_years = set(train_df['year'].unique())
        val_years   = set(val_df['year'].unique())
        test_years  = set(test_df['year'].unique())

        assert len(train_years & val_years)  == 0, "Train and val years overlap"
        assert len(train_years & test_years) == 0, "Train and test years overlap"
        assert len(val_years   & test_years) == 0, "Val and test years overlap"

    def test_train_before_val_before_test(self, features_df):
        """All train years must precede all val years which precede all test years."""
        train_df, val_df, test_df = make_splits(features_df)
        assert train_df['year'].max() < val_df['year'].min(), \
            "Some train years are >= val years"
        assert val_df['year'].max() < test_df['year'].min(), \
            "Some val years are >= test years"

    def test_no_nulls_after_preprocessing(self, features_df):
        """X_train, X_val, X_test must have zero nulls after preprocessing."""
        train_df, val_df, test_df = make_splits(features_df)
        X_train, X_val, X_test, *_ = preprocess(train_df, val_df, test_df)

        assert np.isnan(X_train).sum() == 0, "Nulls remain in X_train"
        assert np.isnan(X_val).sum()   == 0, "Nulls remain in X_val"
        assert np.isnan(X_test).sum()  == 0, "Nulls remain in X_test"

    def test_scaler_fit_on_train_only(self, features_df):
        """
        Verify the fit-on-train rule: continuous features in X_train
        should have mean ≈ 0 and std ≈ 1 after scaling.
        Val/test are NOT expected to be exactly standardised.
        """
        train_df, val_df, test_df = make_splits(features_df)
        X_train, X_val, X_test, *_ = preprocess(train_df, val_df, test_df)

        from src.preprocessing import CONT_IDX
        train_means = X_train[:, CONT_IDX].mean(axis=0)
        train_stds  = X_train[:, CONT_IDX].std(axis=0)

        np.testing.assert_allclose(train_means, 0.0, atol=1e-5,
            err_msg="Train continuous features should have mean ≈ 0 after scaling")
        np.testing.assert_allclose(train_stds,  1.0, atol=1e-5,
            err_msg="Train continuous features should have std ≈ 1 after scaling")

    def test_correct_output_shapes(self, features_df):
        """X arrays must have shape (n, 12) and y arrays shape (n,)."""
        train_df, val_df, test_df = make_splits(features_df)
        X_train, X_val, X_test, y_train, y_val, y_test, *_ = preprocess(
            train_df, val_df, test_df
        )
        assert X_train.shape[1] == len(FEATURE_COLS), \
            f"X_train has {X_train.shape[1]} features, expected {len(FEATURE_COLS)}"
        assert X_train.shape[0] == y_train.shape[0]
        assert X_val.shape[0]   == y_val.shape[0]
        assert X_test.shape[0]  == y_test.shape[0]

    def test_split_inputs_shapes(self, features_df):
        """split_inputs() must return correct shapes for DNN input branches."""
        train_df, val_df, test_df = make_splits(features_df)
        X_train, *_ = preprocess(train_df, val_df, test_df)
        X_cont, X_carrier, X_airport = split_inputs(X_train)

        assert X_cont.shape    == (X_train.shape[0], 10)
        assert X_carrier.shape == (X_train.shape[0],)
        assert X_airport.shape == (X_train.shape[0],)
        assert X_cont.dtype    == np.float32
        assert X_carrier.dtype == np.int32
        assert X_airport.dtype == np.int32


# ══════════════════════════════════════════════════════════════════════
# 7. Label encoding tests
# ══════════════════════════════════════════════════════════════════════

class TestLabelEncoding:

    def test_carrier_enc_non_negative(self, features_df):
        """carrier_enc must be non-negative integers."""
        assert (features_df['carrier_enc'] >= 0).all()

    def test_airport_enc_non_negative(self, features_df):
        """airport_enc must be non-negative integers."""
        assert (features_df['airport_enc'] >= 0).all()

    def test_carrier_enc_count(self, features_df, raw_df):
        """Number of unique carrier_enc values must equal number of unique carriers."""
        n_carriers_raw = raw_df['carrier'].nunique()
        n_carriers_enc = features_df['carrier_enc'].nunique()
        assert n_carriers_enc == n_carriers_raw, \
            f"Expected {n_carriers_raw} carriers, got {n_carriers_enc}"

    def test_consistent_encoding(self, raw_df):
        """
        The same carrier must always map to the same integer — verify by
        encoding twice with the same encoder and comparing.
        """
        df1, le_c, le_a = build_features(raw_df)
        df2, _, _       = build_features(raw_df, le_carrier=le_c, le_airport=le_a)

        pd.testing.assert_series_equal(
            df1['carrier_enc'].reset_index(drop=True),
            df2['carrier_enc'].reset_index(drop=True),
            check_names=False,
        )
