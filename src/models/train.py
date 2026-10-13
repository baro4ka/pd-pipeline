"""Training entry point: load -> feature engineering -> split -> GridSearchCV -> metrics.

Usage:
    python -m src.models.train            # full dataset
    python -m src.models.train --quick    # 5k-row subsample (fast smoke run)

Outputs (data/processed/):
    roc_curve.png, best_logreg.joblib, best_gb.joblib, metrics.csv
"""

from __future__ import annotations

import argparse
import sys

import joblib
import pandas as pd

from ..data.feature_engineering import build_features
from ..data.load import PROCESSED_DIR, TARGET, clean, load_raw
from .pipeline import FEATURES, plot_roc_curve, train


def prepare_data(quick: bool = False) -> pd.DataFrame:
    df = clean(load_raw())
    df = build_features(df)
    if quick:
        df = df.sample(5000, random_state=42)
    return df


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="5k-row subsample (smoke run)")
    args = parser.parse_args(argv)

    df = prepare_data(quick=args.quick)
    X = df[FEATURES]
    y = df[TARGET]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"train={len(X_train)} test={len(X_test)} features={len(FEATURES)}")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("logreg", "gb"):
        search, metrics = train(X_train, y_train, X_test, y_test, model_name=name)
        print(f"[{name}] best_params={search.best_params_}")
        print(f"[{name}] cv_roc_auc={search.best_score_:.4f}  test={metrics}")
        rows.append({"model": name, **metrics, "cv_roc_auc": search.best_score_})

        joblib.dump(search.best_estimator_, PROCESSED_DIR / f"best_{name}.joblib")
        if name == "logreg":
            y_prob = search.best_estimator_.predict_proba(X_test)[:, 1]
            ax = plot_roc_curve(y_test.to_numpy(), y_prob)
            ax.figure.savefig(PROCESSED_DIR / "roc_curve.png", dpi=120)

    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "metrics.csv", index=False)
    print(f"artifacts -> {PROCESSED_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
