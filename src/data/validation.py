"""Валидация входящих данных через Great Expectations (fluent API 1.x).

Сюит pd_raw_suite проверяет сырой датасет: точный набор колонок, пропуски,
типы, домены кодировок/статусов оплаты и диапазоны значений.

Использование:
    from src.data.validation import assert_valid, validate_raw
    result = validate_raw(df)      # результат валидации GE
    assert_valid(df)               # ValueError при несоответствии
"""

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

# Статус оплаты: -2 погашено, -1 долга нет, 0 в срок, 1..8 месяцев просрочки
PAY_STATUS_DOMAIN = [-2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
SEX_DOMAIN = [1, 2]
EDUCATION_DOMAIN = [0, 1, 2, 3, 4, 5, 6]
MARRIAGE_DOMAIN = [0, 1, 2, 3]

INT_COLUMNS = ["SEX", "EDUCATION", "MARRIAGE", "AGE", *PAY_COLS, TARGET]
FLOAT_COLUMNS = ["LIMIT_BAL", *BILL_COLS, *PAYAMT_COLS]

_ctx = None


def _get_context():
    """Общий ephemeral-контекст GE (создаётся один раз на процесс)."""
    global _ctx
    if _ctx is None:
        _ctx = ge.get_context()
        _ctx.variables.progress_bars = {"globally": False}
    return _ctx


def build_raw_suite():
    """Собирает GE-сьюит для сырых данных UCI Credit Card."""
    _get_context()  # add_expectation требует активный data context
    suite = ExpectationSuite(
        name=SUITE_NAME,
        meta={"description": "Валидация входящих сырых данных (UCI Credit Card)"},
    )

    # схема: точный набор колонок
    suite.add_expectation(
        ExpectTableColumnsToMatchSet(column_set=EXPECTED_COLUMNS, exact_match=True)
    )

    # пропуски
    for col in EXPECTED_COLUMNS:
        suite.add_expectation(ExpectColumnValuesToNotBeNull(column=col))

    # типы: коды, статусы и таргет — целые
    for col in INT_COLUMNS:
        suite.add_expectation(ExpectColumnValuesToBeOfType(column=col, type_="int64"))
    # суммы: int64 допустим, т.к. CSV round-trip читает целочисленные колонки как int64
    for col in FLOAT_COLUMNS:
        suite.add_expectation(
            ExpectColumnValuesToBeInTypeList(column=col, type_list=["int64", "float64"])
        )

    # домены значений
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="SEX", value_set=SEX_DOMAIN))
    suite.add_expectation(
        ExpectColumnValuesToBeInSet(column="EDUCATION", value_set=EDUCATION_DOMAIN)
    )
    suite.add_expectation(ExpectColumnValuesToBeInSet(column="MARRIAGE", value_set=MARRIAGE_DOMAIN))
    for col in PAY_COLS:
        suite.add_expectation(ExpectColumnValuesToBeInSet(column=col, value_set=PAY_STATUS_DOMAIN))
    suite.add_expectation(ExpectColumnValuesToBeInSet(column=TARGET, value_set=[0, 1]))

    # диапазоны: фактические границы датасета + запас
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


def validate_raw(df, suite=None):
    """Валидирует df по сьюиту, возвращает результат GE."""
    ctx = _get_context()
    try:
        ds = ctx.data_sources.get("pd_raw")
    except KeyError:
        ds = ctx.data_sources.add_pandas(name="pd_raw")
    batch = ds.read_dataframe(df)  # ephemeral-ассет на каждый вызов
    return batch.validate(suite if suite is not None else build_raw_suite())


def assert_valid(df, suite=None):
    """Валидация с падением ValueError при любом несоответствии."""
    result = validate_raw(df, suite)
    if not result.success:
        failed = [r.expectation_config["kwargs"] for r in result.results if not r.success]
        raise ValueError(
            f"Валидация данных не пройдена: не выполнены {len(failed)} expectation: {failed}"
        )
