"""Prepare stage (DVC): raw CSV -> cleaned + featured dataset.

Usage:
    python -m src.data.prepare

Output: data/processed/features.csv (all engineered features + target).
"""

from __future__ import annotations

import sys

from .feature_engineering import build_features
from .load import PROCESSED_DIR, TARGET, clean, load_raw


def main() -> int:
    df = clean(load_raw())
    df = build_features(df)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out = PROCESSED_DIR / "features.csv"
    df.to_csv(out, index=False)
    print(f"prepared {len(df)} rows x {len(df.columns)} cols -> {out}")
    print(f"target balance: {df[TARGET].mean():.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
