# PD-пайплайн — расширенное описание проекта

> Краткая справка — в `README.md`. Этот документ — детальное описание:
> архитектура, методика, решения, результаты и воспроизводимость.

## Проект

- **Дисциплина:** Автоматизация процессов разработки и тестирования моделей ML
  (магистратура НИЯУ МИФИ / SkillFactory, 3 семестр, Трек 2).
- **Роль:** ML-инженер кредитного отдела банка.
- **Задача:** сквозной автоматизированный пайплайн разработки, тестирования,
  развёртывания и мониторинга скоринговой модели предсказания вероятности дефолта
  клиента (PD-модель).
- **Оценка:** 50 баллов по 9 критериям (Git/структура, данные+валидация, модель,
  MLflow, DVC, тесты+CI, Docker+FastAPI, дрифт, презентация).

## Цели проекта

- Автоматизировать полный цикл: данные → валидация → фичи → обучение → тесты →
  развёртывание → мониторинг.
- Обеспечить воспроизводимость (пининг зависимостей, фиксированные seed'ы,
  DVC-пайплайн, Docker-образ).
- Закрыть требования ТЗ по 9 критериям и задокументировать архитектурные решения.

## Доменные данные

Датасет: [Default of Credit Card Clients (UCI)](https://archive.ics.uci.edu/dataset/350/default-of-credit-card-clients-dataset).

- Файл: `data/raw/UCI_Credit_Card.csv` — 30 000 строк, 25 колонок, 2.9 МБ.
- Таргет: `default.payment.next.month` (0/1) — дефолт в следующем месяце.
- Баланс классов: ~22% дефолтов — умеренный дисбаланс.
- Особенности:
  - отсутствует колонка `PAY_1` (нумерация `PAY_0, PAY_2…PAY_6`) — учтено в
    feature engineering (`PAY_COLS` собран по факту) и в валидации (точный список колонок);
  - `PAY_*` — статус платежа: `-2` без записи, `-1` в срок, `0…8` — месяцы просрочки;
  - `BILL_AMT*` / `PAY_AMT*` — суммы счетов и предыдущих платежей (могут быть отрицательными).

## Архитектура

```
UCI_Credit_Card.csv (data/raw)
        │  load.py + clean()
        ▼
Great Expectations: валидация (73 expectations)     ← CI-шаг, exit 1 на аномалиях
        │  feature_engineering.py (17 инженерных фич)
        ▼
data/processed/features.csv (22 признака)           ← DVC-стадия prepare
        │  train.py: sklearn Pipeline + GridSearchCV
        ▼
models/best_model.joblib + metrics.csv              ← DVC-стадия train
        │
        ├── MLflow: params/metrics/артефакты (mlruns/)
        ├── FastAPI /predict (Docker, :8000)
        └── drift.py: «новые» данные → API → PSI
```

Разделение на стадии `prepare`/`train` (DVC) позволяет переобучать модель без
пересчёта фич: если код `src/data/*` и датасет не менялись, стадия `prepare`
пропускается (хэши в `dvc.lock` совпадают).

## Валидация данных (Great Expectations)

Suite `pd_raw_suite` — **73 expectations** (`src/data/validation.py`):

| Группа | Кол-во | Примеры |
|---|---|---|
| Структура | 1 | точный список из 25 колонок |
| Null-проверки | 25 | каждая колонка без null |
| Типы | 23 | int64 для кодировок, `[int64, float64]` для суммовых |
| Домены | 10 | SEX ∈ {0,1,2}, EDUCATION ∈ {0..6}, PAY ∈ {-2..8}, TARGET ∈ {0,1} |
| Диапазоны | 14 | BILL_AMT: −339 603…1 664 089, AGE: 21..69 и т.д. |

Диапазоны подобраны по фактическим данным датасета — suite валидирует реальность,
а не «красивые» границы.

**Аномальные тесты** (`tests/test_validation.py`, 6 шт.): PAY=99, AGE=200, NaN,
удалённая колонка, SEX=7, float вместо int — каждый проверяет `not result.success`,
т.е. suite обязан падать на аномалии (прямое требование ТЗ).

CLI-точка входа для CI: `python -m src.data.validate_data [path.csv]` (exit 1 на аномалиях).

Техническое решение: GE 1.24.0 имеет баг в `validate()` с DataFrame напрямую
(`any([batch_data, ...])` падает на булеве DataFrame) — используется официальный
fluent API: `ctx.data_sources.add_pandas()` → `ds.read_dataframe(df)` →
`batch.validate(suite)`. Публичный API без monkey-patch'ей.

## Feature engineering

17 инженерных признаков (`src/data/feature_engineering.py`), итого в модели 22 признака
(5 исходных + 17):

| Группа | Признаки |
|---|---|
| Агрегаты по 6-месячной истории платежей | `pay_max_status`, `pay_min_status`, `pay_mean_status`, `pay_std_status` |
| Задержки | `pay_num_late` (>0), `pay_num_severe` (≥2), `pay_num_norecord` (=-2), `pay_consec_late` (подряд со свежей) |
| Тренды | `pay_trend` (PAY_0 − PAY_6), `bill_trend` (BILL_AMT1 − BILL_AMT6) |
| Суммы | `bill_mean`, `bill_max`, `payamt_mean`, `payamt_max`, `pay_to_bill` (PAY_AMT1/BILL_AMT1) |
| Прочее | `age_group` (биннинг возраста), `limit_bal_log` (log1p лимита) |

Методика — агрегация по временному ряду платежей (вебинар 09.09, лекции 3/4 курса).

## Модель

`src/models/pipeline.py` — один sklearn Pipeline на всю предобработку (защита от
data leakage: импутация/масштабирование/кодирование вписаны в fit):

- numeric: `SimpleImputer(median)` + `StandardScaler`;
- categorical: `astype(str)` + `SimpleImputer(constant)` +
  `OneHotEncoder(handle_unknown='ignore')`.

Обучение (`src/models/train.py`): **5 экспериментов** (требование ТЗ «≥5»):

1. `logreg_grid` — LogisticRegression (class_weight=balanced), GridSearchCV по C;
2. `gb_grid` — GradientBoostingClassifier, GridSearchCV (lr/depth/n_estimators);
3. `gb_shallow` — GradientBoosting, фиксированный «плоский» конфиг;
4. `hgb` — HistGradientBoostingClassifier, фиксированный конфиг;
5. `rf` — RandomForestClassifier, фиксированный конфиг.

GridSearchCV: `scoring=roc_auc, cv=5`, split 80/20 stratified. `n_jobs=2`
(на 16 ГБ RAM `n_jobs=-1` вызывал OOM в joblib.dump).

### Результаты (датасет 30 000, test ROC-AUC)

| Эксперимент | Best params | CV ROC-AUC | Test ROC-AUC | P | R | F1 |
|---|---|---|---|---|---|---|
| **gb_grid** | lr=0.1, depth=3, n=200 | 0.7832 | **0.7761** | 0.646 | 0.376 | 0.475 |
| gb_shallow | — | — | 0.7745 | 0.646 | 0.374 | 0.474 |
| hgb | — | — | 0.7738 | 0.642 | 0.379 | 0.477 |
| rf | — | — | 0.7728 | 0.453 | 0.619 | 0.523 |
| logreg_grid | C=10 | 0.7719 | 0.7574 | 0.438 | 0.613 | 0.511 |

Лучшая модель: **gb_grid** (test ROC-AUC 0.7761), сериализована в
`models/best_model.joblib` (joblib, 274 КБ). ROC-кривые: `data/processed/roc_curve.png`
(общая) + per-experiment PNG. Таблица метрик: `data/processed/metrics.csv`.

## MLflow

Каждый эксперимент логирует в `mlruns/` (file store):

- **params:** модель, гиперпараметры, n_train/n_test/n_features;
- **metrics:** test_roc_auc/precision/recall/f1, cv_roc_auc;
- **артефакты:** ROC PNG + модель через `mlflow.sklearn.log_model(name='model')`.

Фактически: 10 FINISHED-ранов (2 полных прогона × 5 экспериментов).
UI: `mlflow ui --backend-store-uri file:mlruns`.

Особенность MLflow 3.x: `artifact_path` в `log_model` deprecated, модель уходит в
model store, привязанный к рану — проверено, что модель видна по рану через client.

## DVC

- `dvc.yaml` — 2 стадии: `prepare` (load+FE → features.csv) и `train`
  (train.py → best_model.joblib + metrics.csv);
- `dvc.lock` — хэши (md5) зависимостей и артефактов; `dvc repro` — воспроизведение
  (неизменённые стадии пропускаются);
- датасет, подготовленные данные и модель хранятся **в git** (репозиторий) —
  DVC-remote не используется: современные версии DVC (2.x и 3.x) не поддерживают
  git-репозиторий как remote (требуется облачное хранилище), а размер данных
  (~10 МБ) позволяет держать их в git. DVC остаётся инструментом
  воспроизводимости пайплайна.

## Тесты

**21 тест** в 3 файлах (`tests/`), 0 провалов:

| Файл | Тестов | Что проверяет |
|---|---|---|
| `test_validation.py` | 8 | GE-suite: валидные данные проходят, 6 аномалий ловятся |
| `test_pipeline.py` | 8 | Pipeline: формы данных, отсутствие leakage, предсказания, метрики |
| `test_api.py` | 5 | FastAPI: /health, /predict (одна/батч/недостающие фичи/пусто) |

Линтеры: flake8 (`.flake8`, line-length 100) + black (`pyproject.toml`).

## CI (GitHub Actions)

`.github/workflows/ci.yml` — на каждый push/PR:

```
flake8 src tests → black --check src tests →
python -m src.data.validate_data → pytest -v
```

Python 3.12, зависимости из `requirements.txt` (все версии зафиксированы).

## Docker + FastAPI

`Dockerfile`: python:3.12-slim → requirements → COPY src + models/best_model.joblib →
uvicorn :8000.

### `GET /health`

```json
{ "status": "ok", "model": "best_model.joblib", "n_features": 22 }
```

### `POST /predict`

Одна запись или батч (массив) из 22 признаков.

**Пример запроса (curl):**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000, "AGE": 34, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 1, "PAY_4": -1, "PAY_5": 0, "PAY_6": 0,
    "pay_max_status": 2, "pay_min_status": -1, "pay_mean_status": 0.67,
    "pay_std_status": 1.2, "pay_num_late": 3, "pay_num_severe": 2,
    "pay_num_norecord": 0, "pay_trend": 2, "pay_consec_late": 2,
    "bill_mean": 50000, "bill_max": 80000, "bill_trend": 10000,
    "payamt_mean": 20000, "payamt_max": 40000, "pay_to_bill": 0.4,
    "age_group": 1, "limit_bal_log": 12.2
  }'
```

**Ответ:**

```json
{ "default_probability": 0.4236, "class": 0 }
```

- батч → `{"predictions": [ {...}, ... ]}`;
- 422 — на недостающих признаках или пустом теле.

Скрипты: `scripts/build_and_run.{ps1,sh}` — сборка образа, запуск контейнера,
автотесты /health и /predict (ожидание готовности — polling до 60 с).

## Мониторинг дрифта (F8)

`src/monitoring/drift.py`: имитация «новых» данных (1000 строк — слайс тестовой
выборки) → POST /predict → **PSI** (Population Stability Index):

- 10 бинов, порог 0.2 (статус ALERT);
- считается по вероятности дефолта и 6 ключевым признакам;
- baseline — те же данные через тот же API (честное сравнение распределений).

Результат на одном распределении (ожидаемо): ПСИ 0.003–0.023 (ok),
средняя вероятность 0.2186 → 0.2236.

## Воспроизводимость

- все зависимости зафиксированы в `requirements.txt` (pin'ы);
- фиксированные seed'ы (split, GridSearchCV, сабсэмпл);
- `dvc repro` — воспроизведение артефактов по хэши;
- Docker-образ собирается из Dockerfile + requirements;
- датасет и артефакты в git (хэши в `dvc.lock` совпадают с файлами).

## Запуск (полный список)

```bash
pip install -r requirements.txt

# EDA
jupyter notebook notebooks/01_eda.ipynb

# Валидация данных (CI-шаг)
python -m src.data.validate_data            # data/raw/UCI_Credit_Card.csv
python -m src.data.validate_data path.csv   # произвольный файл

# Тесты
pytest

# Обучение (5 экспериментов в MLflow)
python -m src.models.train           # полный датасет (~15-20 мин)
python -m src.models.train --subsample  # сабсэмпл 5k (быстрая проверка)
python -m src.models.train --no-mlflow  # без логирования

# MLflow UI
mlflow ui --backend-store-uri file:mlruns

# API локально
uvicorn src.api.app:app --port 8000

# Docker: сборка + запуск + автотесты
.\scripts\build_and_run.ps1        # Windows
./scripts/build_and_run.sh         # Linux/macOS

# Мониторинг дрифта (API должен быть запущен)
python -m src.monitoring.drift --api http://localhost:8000 --n 1000

# DVC-пайплайн
dvc repro
```

## Итоговые цифры

| Показатель | Значение |
|---|---|
| Коммиты | 8 milestone-коммитов (F0-F1…F13), working tree чистый |
| Тесты | 21 (3 файла), 0 провалов; CI зелёный |
| GE expectations | 73; аномальных тестов — 6 |
| Инженерных фич | 17 (в модели 22 признака) |
| MLflow | 10 ранов FINISHED, 5 экспериментов/прогон |
| Лучшая модель | gb_grid: test ROC-AUC 0.7761 (CV 0.7832) |
| DVC | 2 стадии, dvc.lock, `dvc repro` — up to date |
| CI | 4 шага (flake8, black, GE, pytest) |
| API | /predict: класс + вероятность, батчи, 422 |
| Дрифт | ПСИ 0.003–0.023 (ok) |
| Презентация | 15 слайдов |

## Ограничения и перспективы

- **Калибровка вероятностей** (Platt/isotonic) — PD-модель в банке требует
  откалиброванных вероятностей; сейчас `default_probability` — raw-оценка.
- **Данные одного срезa** — датасет UCI (2005-2006, Тайвань); в production нужна
  свежая история и временной split.
- **Дрифт** — имитация на слайсе тестовой выборки; в production — реальный
  поток заявок и алерты по ПСИ/расхождениям распределений.
- **A/B-тестирование** моделей (паттерн из проекта 2 семестра «Внедрение ML»)
  как следующий шаг после развёртывания.
