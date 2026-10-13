"""Data loading and initial cleaning for the PD (default prediction) model."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CSV = DATA_DIR / "raw" / "UCI_Credit_Card.csv"
PROCESSED_DIR = DATA_DIR / "processed"

TARGET = "default.payment.next.month"

# Payment-status columns, ordered most recent (PAY_0) -> oldest (PAY_6)
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_BASE = ["LIMIT_BAL", "AGE"]


def load_raw(path=RAW_CSV) -> pd.DataFrame:
    """Load the raw UCI credit-card dataset."""
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Initial cleaning: drop ID, drop rows with any missing value, validate target.

    Returns a copy. The target column keeps its original name (see TARGET).
    """
    df = df.drop(columns=["ID"], errors="ignore").copy()
    df = df.dropna()
    bad = df[~df[TARGET].isin([0, 1])]
    if len(bad):
        raise ValueError(f"Target {TARGET} has non-binary values: {bad[TARGET].unique()}")
    return df
