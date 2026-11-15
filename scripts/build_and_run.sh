#!/usr/bin/env bash
# Сборка Docker-образа и локальный запуск API-контейнера.
set -euo pipefail
cd "$(dirname "$0")/.."

docker build -t pd-model:latest .

docker rm -f pd-api 2>/dev/null || true
docker run -d --name pd-api -p 8000:8000 pd-model:latest

sleep 8
echo "--- /health ---"
curl -s http://localhost:8000/health
echo
echo "--- /predict (sample) ---"
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "LIMIT_BAL": 200000, "AGE": 34, "SEX": 1, "EDUCATION": 2, "MARRIAGE": 1,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 1, "PAY_4": -1, "PAY_5": 0, "PAY_6": 0,
    "pay_max_status": 2, "pay_min_status": -1, "pay_mean_status": 0.67, "pay_std_status": 1.2,
    "pay_num_late": 3, "pay_num_severe": 2, "pay_num_norecord": 0, "pay_trend": 2,
    "pay_consec_late": 2, "bill_mean": 50000, "bill_max": 80000, "bill_trend": 10000,
    "payamt_mean": 20000, "payamt_max": 40000, "pay_to_bill": 0.4,
    "age_group": 1, "limit_bal_log": 12.2
  }'
echo
echo "API запущен: http://localhost:8000 (контейнер: pd-api)"
echo "Остановка: docker rm -f pd-api"
