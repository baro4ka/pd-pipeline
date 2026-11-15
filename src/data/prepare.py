"""Стадия DVC prepare: сырой CSV -> очищенный датасет с инженерными признаками.

Запуск: python -m src.data.prepare
Результат: data/processed/features.csv (все признаки + таргет).
"""

import sys

from .feature_engineering import build_features
from .load import PROCESSED_DIR, TARGET, clean, load_raw


def main():
    df = clean(load_raw())
    df = build_features(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features.csv"
    df.to_csv(out, index=False)
    print(f"подготовлено {len(df)} строк x {len(df.columns)} колонок -> {out}")
    print(f"доля дефолтов: {df[TARGET].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
