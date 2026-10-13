"""CLI: validate the raw dataset with the Great Expectations suite.

Usage:
    python -m src.data.validate_data [path-to-csv]

Exits 0 if the data passes the suite, 1 otherwise (for CI usage).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .load import RAW_CSV
from .validation import assert_valid


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else RAW_CSV
    df = pd.read_csv(path)
    try:
        assert_valid(df)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"OK: {len(df)} rows passed the pd_raw_suite validation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
