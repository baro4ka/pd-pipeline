"""Data validation for the PD pipeline using Great Expectations (1.x fluent API).

The suite ``pd_raw_suite`` checks the incoming raw dataset for:
  * the expected set of columns (exact match),
  * missing values (nulls) in every column,
  * data types (integer vs float columns),
  * value domains (categorical codes, payment status) and ranges
    (age, credit limit, bill/payment amounts).

Usage:
    from src.data.validation import assert_valid, validate_raw
    result = validate_raw(df)      # ExpectationSuiteValidationResult
    assert_valid(df)               # raises ValueError on failure
"""

from __future__ import annotations

import pandas as pd
import great_expectations as ge
from great_expectations.core import ExpectationSuite
from great_expectations.expectations import (
    ExpectColumnValuesToBeBetween,
    ExpectColumnValuesToBeInSet,
    ExpectColumnValuesToBeInTypeList,
    ExpectColumnValuesToBeOfType,
    ExpectColumnValuesToNotBeNull,
    ExpectTableColumnsToMatchSet,
)

from .load import BILL_COLS, PAY_COLS, PAYAMT_COLS, TARGET

SUITE_NAME = "pd_raw_suite"

EXPECTED_COLUMNS = [
    "ID",
    "LIMIT_BAL",
    "SEX",
    "EDUCATION",
    "MARRIAGE",
    "AGE",
    "PAY_0",
    "PAY_2",
    "PAY_3",
    "PAY_4",
    "PAY_5",
    "PAY_6",
    *BILL_COLS,
    *PAYAMT_COLS,
    TARGET,
]

# Payment status: -2 paid off, -1 no debt, 0 paid on time, 1..8 months delayed.
PAY_STATUS_DOMAIN = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
SEX_DOMAIN = [1, 2]
EDUCATION_DOMAIN = [0, 1, 2, 3, 4, 5, 6]
MARRIAGE_DOMAIN = [0, 1, 2, 3]

INT_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_COLS, TARGET]
FLOAT_COLUMNS = ["LIMIT_BAL", *BILL_COLS, *PAYAMT_COLS]

_ctx: "ge.data_context.data_context.abstract_data_context.AbstractDataContext | None" = None


def _get_context():
    """Shared ephemeral data context (created lazily once per process)."""
    global _ctx
    if _ctx is None:
        _ctx = ge.get_context()
        _ctx.variables.progress_bars = {"globally": False}
    return _ctx


def build_raw_suite() -> ExpectationSuite:
    """Build the Great Expectations suite for the raw UCI credit-card data."""
    _get_context()  # add_expectation requires an active data context
    suite = ExpectationSuite(
        name=SUITE_NAME,
        meta={"description": "Incoming raw data validation (UCI Credit Card)"},
    )

    # --- Schema: exact column set ---
    suite.add_expectation(
        ExpectTableColumnsToMatchSet(column_set=EXPECTED_COLUMNS, exact_match=True)
    )

    # --- Nulls ---
    for col in EXPECTED_COLUMNS:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

    # --- Data types ---
    # Categorical codes, payment status and the target must be integers.
    for col in INT_COLUMNS:
        suite.add_expectation(ExpectColumnValuesToBeOfType(column=col, type_="int64"))
    # Amount columns are numeric; int64 is allowed because a CSV round-trip
    # reads all-integer columns as int64.
    for col in FLOAT_COLUMNS:
        suite.add_expectation(
            ExpectColumnValuesToBeInTypeList(column=col, type_list=["int64", "float64"])
        )

    # --- Value domains ---
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="SEX", value_set=SEX_DOMAIN))
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="EDUCATION", value_set=EDUCATION_DOMAIN)
    )
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="MARRIAGE", value_set=MARRIAGE_DOMAIN))
    for col in PAY_COLS:
        suite.add_expectation(ExpectColumnValuesToBeInSet(column=col, value_set=PAY_STATUS_DOMAIN))
    suite.add_expectation(ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1]))

    # --- Ranges (observed dataset bounds with safety margin) ---
    suite.add_expectation(ExpectColumnValuesToBeBetween(column="AGE", min_value=21, max_value=80))
    suite.add_expectation(
        ExpectColumnValuesToBeBetween(column="LIMIT_BAL", min_value=0, max_value=2_000_000)
    )
    for col in BILL_COLS:
        suite.add_expectation(
            ExpectColumnValuesToBeBetween(column=col, min_value=-500_000, max_value=2_000_000)
        )
    for col in PAYAMT_COLS:
        suite.add_expectation(
            ExpectColumnValuesToBeBetween(column=col, min_value=0, max_value=2_000_000)
        )

    return suite


def validate_raw(
    df: pd.DataFrame, suite: ExpectationSuite | None = None
) -> "ge.core.expectation_validation_result.ExpectationSuiteValidationResult":
    """Validate ``df`` against the raw-data suite and return the result."""
    ctx = _get_context()
    try:
        ds = ctx.data_sources.get("pd_raw")
    except KeyError:
        ds = ctx.data_sources.add_pandas(name="pd_raw")
    batch = ds.read_dataframe(df)  # ephemeral asset per call
    return batch.validate(suite if suite is not None else build_raw_suite())


def assert_valid(df: pd.DataFrame, suite: ExpectationSuite | None = None) -> None:
    """Validate ``df`` and raise ``ValueError`` if any expectation fails."""
    result = validate_raw(df, suite)
    if not result.success:
        failed = [r.expectation_config["kwargs"] for r in result.results if not r.success]
        raise ValueError(f"Data validation failed: {len(failed)} expectation(s) not met: {failed}")
