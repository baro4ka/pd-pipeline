"""Мониторинг дрифта данных (F8): имитация поступления новых данных + PSI.

Берёт срез тестовой выборки (имитация «новых» данных), отправляет на работающий
API и сравнивает распределения с тренировочными через Population Stability Index
(PSI) по вероятности дефолта и ключевым признакам.

Запуск: python -m src.monitoring.drift --api http://localhost:8000 --n 1000
"""

import argparse
import sys

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

from ..data.feature_engineering import build_features
from ..data.load import TARGET, clean, load_raw
from ..models.pipeline import FEATURES

PSI_BINS = 10
PSI_THRESHOLD = 0.2


def compute_psi(expected, actual, bins=PSI_BINS):
    """Population Stability Index между двумя числовыми распределениями."""
    quantiles = expected.quantile(np.linspace(0, 1, bins + 1)).unique()
    if len(quantiles) < 3:  # вырожденное распределение
        return 0.0
    cats_e = pd.cut(expected, bins=quantiles, include_lowest=True)
    cats_a = pd.cut(actual, bins=quantiles, include_lowest=True)
    labels = cats_e.cat.categories
    pe = cats_e.value_counts(normalize=True).reindex(labels, fill_value=0)
    pa = cats_a.value_counts(normalize=True).reindex(labels, fill_value=0)
    pe = pe.clip(lower=1e-4)
    pa = pa.clip(lower=1e-4)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def main(argv=None):
    parser = argparse.ArgumentParser(description="Мониторинг дрифта: PSI по API-предсказаниям")
    parser.add_argument("--api", default="http://localhost:8000", help="базовый URL API")
    parser.add_argument(
        "--n", type=int, default=1000, help="сколько строк «новых» данных отправить"
    )
    args = parser.parse_args(argv)

    df = clean(load_raw())
    df = build_features(df)

    train_part, test_part = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df[TARGET]
    )
    new_data = test_part.sample(min(args.n, len(test_part)), random_state=42)

    resp = requests.post(
        f"{args.api}/predict", json=new_data[FEATURES].to_dict(orient="records"), timeout=120
    )
    resp.raise_for_status()
    preds = pd.DataFrame(resp.json()["predictions"])
    new_data = new_data.copy()
    new_data["default_probability"] = preds["default_probability"].to_numpy()

    # baseline: тренировочный срез через тот же API, чтобы сравнение было честным
    train_sample = train_part.sample(min(args.n, len(train_part)), random_state=42)
    resp = requests.post(
        f"{args.api}/predict", json=train_sample[FEATURES].to_dict(orient="records"), timeout=120
    )
    resp.raise_for_status()
    train_sample = train_sample.copy()
    train_sample["default_probability"] = pd.DataFrame(resp.json()["predictions"])[
        "default_probability"
    ].to_numpy()

    key_features = [
        "LIMIT_BAL",
        "AGE",
        "PAY_0",
        "pay_max_status",
        "pay_consec_late",
        "limit_bal_log",
    ]
    for col in ["default_probability"] + key_features:
        psi = compute_psi(train_sample[col], new_data[col])
        status = "ALERT" if psi > PSI_THRESHOLD else "ok"
        print(f"ПСИ {col:20s} = {psi:.4f}  ({status})")

    print(f"baseline (train): средняя вероятность {train_sample['default_probability'].mean():.4f}")
    print(f"входящие (new):   средняя вероятность {new_data['default_probability'].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
