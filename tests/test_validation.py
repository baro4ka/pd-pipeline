"""Tests for the Great Expectations validation suite (F2).

The main test validates the real incoming dataset. The anomaly tests inject
defects into a sample and assert that the suite catches them — this is the
"test that fails if anomalies appear in new data" required by the TZ.
"""

import numpy as np
import pandas as pd
import pytest

from src.data.load import RAW_CSV, load_raw
from src.data.validation import assert_valid, build_raw_suite, validate_raw


@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    return load_raw(RAW_CSV)


@pytest.fixture(scope="module")
def suite():
    return build_raw_suite()


@pytest.fixture(scope="module")
def sample(raw_df) -> pd.DataFrame:
    return raw_df.sample(500, random_state=42)


def test_incoming_data_passes_validation(raw_df, suite):
    result = validate_raw(raw_df, suite)
    assert result.success
    assert result.statistics["unsuccessful_expectations"] == 0


def test_assert_valid_passes_on_clean_sample(sample, suite):
    assert_valid(sample, suite)


def test_anomaly_bad_pay_status_fails(sample, suite):
    bad = sample.copy()
    bad.loc[bad.index[0], "PAY_0"] = 99
    result = validate_raw(bad, suite)
    assert not result.success


def test_anomaly_age_out_of_range_fails(sample, suite):
    bad = sample.copy()
    bad.loc[bad.index[0], "AGE"] = 200
    result = validate_raw(bad, suite)
    assert not result.success


def test_anomaly_null_fails(sample, suite):
    bad = sample.copy()
    bad.loc[bad.index[0], "LIMIT_BAL"] = np.nan
    result = validate_raw(bad, suite)
    assert not result.success


def test_anomaly_missing_column_fails(sample, suite):
    bad = sample.drop(columns=["AGE"])
    result = validate_raw(bad, suite)
    assert not result.success


def test_anomaly_bad_categorical_fails(sample, suite):
    bad = sample.copy()
    bad.loc[bad.index[0], "SEX"] = 7
    result = validate_raw(bad, suite)
    assert not result.success


def test_anomaly_wrong_type_fails(sample, suite):
    bad = sample.copy()
    bad["AGE"] = bad["AGE"].astype(float)
    result = validate_raw(bad, suite)
    assert not result.success
