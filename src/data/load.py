"""Загрузка и первичная очистка данных для PD-модели (предсказание дефолта)."""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
RAW_CSV = DATA_DIR / "raw" / "UCI_Credit_Card.csv"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = Path(__file__).resolve().parents[2] / "models"

TARGET = "default.payment.next.month"

# Статусы оплаты: от свежей (PAY_0) к старой (PAY_6); PAY_1 в датасете отсутствует
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAYAMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
CATEGORICAL_COLS = ["SEX", "EDUCATION", "MARRIAGE"]
NUMERIC_BASE = ["LIMIT_BAL", "AGE"]


def load_raw(path=RAW_CSV):
    """Читает исходный CSV (UCI Credit Card)."""
    return pd.read_csv(path)


def clean(df):
    """Первичная очистка: убирает ID и строки с пропусками, проверяет бинарность таргета.

    Возвращает копию; имя таргета не меняется (см. TARGET).
    """
    df = df.drop(columns=["ID"], errors="ignore").copy()
    df = df.dropna()
    bad = df[~df[TARGET].isin([0, 1])]
    if len(bad):
        raise ValueError(f"Таргет {TARGET} содержит небинарные значения: {bad[TARGET].unique()}")
    return df
