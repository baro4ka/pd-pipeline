# PD-пайплайн: модель предсказания дефолта (MLOps)

Сквозной автоматизированный пайплайн разработки, тестирования, развёртывания и
мониторинга скоринговой модели предсказания вероятности дефолта клиента (PD-модель).

- **Дисциплина:** Автоматизация процессов разработки и тестирования моделей ML (магистратура, 3 семестр)
- **Домен:** финансы / кредитный скоринг
- **Датасет:** [Default of Credit Card Clients](https://archive.ics.uci.edu/dataset/350/default-of-credit-card-clients-dataset) (UCI) — 30 000 клиентов, таргет `default.payment.next.month`

## Структура

```
pd-pipeline/
├── data/
│   ├── raw/          # исходный UCI_Credit_Card.csv (управляется DVC)
│   └── processed/    # подготовленные данные
├── notebooks/        # EDA и эксперименты
│   └── 01_eda.ipynb
├── src/
│   ├── data/         # загрузка, очистка, feature engineering, валидация
│   │   ├── load.py
│   │   ├── feature_engineering.py
│   │   └── validation.py      # Great Expectations (F2)
│   ├── models/       # sklearn Pipeline, обучение, MLflow
│   │   ├── pipeline.py        # (F3)
│   │   └── train.py           # (F4)
│   └── api/          # FastAPI /predict (F7)
│       └── app.py
├── tests/            # pytest (F6)
├── models/           # обученные модели (DVC)
├── dvc.yaml          # DVC-пайплайн prepare/train (F5)
├── requirements.txt
└── README.md
```

## Публикация в GitHub (CI + DVC remote)

```bash
git remote add origin https://github.com/<user>/pd-pipeline.git
git push -u origin main

# DVC remote — тот же репозиторий (публичный, pull в CI без токена)
dvc remote add -d origin https://github.com/<user>/pd-pipeline.git
dvc push
```

После пуша GitHub Actions (`.github/workflows/ci.yml`) автоматически:
flake8 → black --check → dvc pull → валидация Great Expectations → pytest.

## Установка

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Данные: `data/raw/UCI_Credit_Card.csv` (после `dvc pull` на этапе F5;
локально файл уже на месте).

## Запуск

```bash
# EDA
jupyter notebook notebooks/01_eda.ipynb

# Валидация данных (Great Expectations, CI-шаг)
python -m src.data.validate_data            # data/raw/UCI_Credit_Card.csv
python -m src.data.validate_data path.csv   # произвольный файл; exit 1 при аномалиях

# Тесты (F6)
pytest

# Обучение (5 экспериментов в MLflow: logreg/gb GridSearchCV + gb/hgb/rf фикс. конфиги)
python -m src.models.train           # полный датасет; артефакты в data/processed/ и mlruns/
python -m src.models.train --subsample  # сабсэмпл 5k (быстрая проверка)

# MLflow UI (эксперименты, метрики, модели, ROC-кривые)
mlflow ui --backend-store-uri file:mlruns

# API локально (F7)
uvicorn src.api.app:app --port 8000

# Docker: сборка образа + запуск контейнера + проверка /health и /predict
.\scripts\build_and_run.ps1        # Windows
./scripts/build_and_run.sh         # Linux/macOS

# Мониторинг дрифта (F8): «новые» данные (часть тестовой) -> API -> PSI
python -m src.monitoring.drift --api http://localhost:8000 --n 1000

# DVC-пайплайн (prepare -> train)
dvc repro
```

## Текущие результаты (F3, датасет 30 000, split 80/20 stratified)

| Модель | Best params (GridSearchCV, scoring=roc_auc, cv=5) | CV ROC-AUC | Test ROC-AUC | P | R | F1 |
|--------|---------------------------------------------------|-----------|--------------|------|------|------|
| LogisticRegression (class_weight=balanced) | C=10 | 0.772 | 0.757 | 0.438 | 0.613 | 0.511 |
| GradientBoostingClassifier | lr=0.1, depth=3, n=200 | 0.783 | 0.776 | 0.646 | 0.376 | 0.475 |

ROC-кривая: `data/processed/roc_curve.png`, таблица метрик: `data/processed/metrics.csv`.

## Этапы (по критериям ТЗ)

| # | Этап | Статус |
|---|------|--------|
| 1 | Git + структура + README | ✅ |
| 2 | EDA, Feature Engineering, валидация (GE/Pandera) | ✅ |
| 3 | Sklearn Pipeline + GridSearchCV + метрики | ✅ |
| 4 | MLflow (≥5 экспериментов) | ✅ |
| 5 | DVC (dvc.yaml, prepare/train) | ✅ |
| 6 | pytest + GitHub Actions | ✅ (workflow готов, запуск после пуша в GitHub) |
| 7 | Docker + FastAPI /predict | ✅ |
| 8 | Мониторинг дрифта (PSI) | ✅ |
| 9 | Презентация | ✅ |
