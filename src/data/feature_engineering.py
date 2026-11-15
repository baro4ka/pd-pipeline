"""Инженерные признаки для PD-модели: агрегаты по 6-месячной истории платежей,
суммы счетов/платежей, биннинг возраста, log кредитного лимита.

Контекст и обоснование — в notebooks/01_eda.ipynb.
"""

import numpy as np
import pandas as pd

from .load import PAY_COLS, BILL_COLS, PAYAMT_COLS


def _consecutive_late(pay):
    """Число месяцев подряд с задержкой (>0), начиная со свежей (PAY_0)."""
    vals = pay.values  # колонки отсортированы: свежая -> старая
    out = np.zeros(vals.shape[0], dtype=int)
    for i in range(vals.shape[0]):
        c = 0
        for j in range(vals.shape[1]):
            if vals[i, j] > 0:
                c += 1
            else:
                break
        out[i] = c
    return pd.Series(out, index=pay.index, name="pay_consec_late")


def build_features(df):
    """Возвращает копию df с добавленными инженерными признаками."""
    df = df.copy()
    pay = df[PAY_COLS]

    # агрегаты по 6-месячной истории платежей
    df["pay_max_status"] = pay.max(axis=1)  # худшая просрочка (0..8)
    df["pay_min_status"] = pay.min(axis=1)
    df["pay_mean_status"] = pay.mean(axis=1)
    df["pay_std_status"] = pay.std(axis=1).fillna(0.0)
    df["pay_num_late"] = (pay > 0).sum(axis=1)  # месяцев с любой задержкой
    df["pay_num_severe"] = (pay >= 2).sum(axis=1)  # месяцев с задержкой >= 2
    df["pay_num_norecord"] = (pay == -2).sum(axis=1)  # месяцев без записи
    df["pay_trend"] = df["PAY_0"] - df["PAY_6"]  # >0: ситуация ухудшается
    df["pay_consec_late"] = _consecutive_late(pay)

    # суммы счетов и предыдущих платежей
    bill = df[BILL_COLS]
    payamt = df[PAYAMT_COLS]
    df["bill_mean"] = bill.mean(axis=1)
    df["bill_max"] = bill.max(axis=1)
    df["bill_trend"] = df["BILL_AMT1"] - df["BILL_AMT6"]  # свежий счёт vs старый
    df["payamt_mean"] = payamt.mean(axis=1)
    df["payamt_max"] = payamt.max(axis=1)
    df["pay_to_bill"] = np.where(df["BILL_AMT1"] > 0, df["PAY_AMT1"] / df["BILL_AMT1"], 0.0)

    # биннинг возраста
    df["age_group"] = pd.cut(
        df["AGE"], bins=[0, 25, 35, 45, 55, 200], labels=[0, 1, 2, 3, 4]
    ).astype(int)

    # log кредитного лимита
    df["limit_bal_log"] = np.log1p(df["LIMIT_BAL"].clip(lower=0))

    return df


FEATURED_COLS = [
    "pay_max_status",
    "pay_min_status",
    "pay_mean_status",
    "pay_std_status",
    "pay_num_late",
    "pay_num_severe",
    "pay_num_norecord",
    "pay_trend",
    "pay_consec_late",
    "bill_mean",
    "bill_max",
    "bill_trend",
    "payamt_mean",
    "payamt_max",
    "pay_to_bill",
    "age_group",
    "limit_bal_log",
]
