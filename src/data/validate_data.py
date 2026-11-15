"""CLI: валидация сырого датасета GE-сьюитом.

Запуск: python -m src.data.validate_data [путь-к-csv]
Код выхода: 0 — данные в норме, 1 — есть аномалии (для CI).
"""

import sys
from pathlib import Path

import pandas as pd

from .load import RAW_CSV
from .validation import assert_valid


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    path = Path(args[0]) if args else RAW_CSV
    df = pd.read_csv(path)
    try:
        assert_valid(df)
    except ValueError as exc:
        print(f"FAIL: {exc}")
        return 1
    print(f"OK: {len(df)} строк прошли валидацию (pd_raw_suite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
