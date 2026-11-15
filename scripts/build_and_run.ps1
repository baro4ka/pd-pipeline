# Сборка Docker-образа и локальный запуск API-контейнера.
# Запуск: .\scripts\build_and_run.ps1

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

docker build -t pd-model:latest .
if ($LASTEXITCODE -ne 0) { throw "сборка образа не удалась" }

docker rm -f pd-api 2>$null
docker run -d --name pd-api -p 8000:8000 pd-model:latest
if ($LASTEXITCODE -ne 0) { throw "запуск контейнера не удался" }

# ждём, пока uvicorn поднимет приложение и загрузит модель (до 60 с)
$ok = $false
for ($i = 0; $i -lt 30; $i++) {
    Start-Sleep -Seconds 2
    try {
        Invoke-RestMethod http://localhost:8000/health -ErrorAction Stop | Out-Null
        $ok = $true
        break
    } catch {}
}
if (-not $ok) { throw "API не ответил за 60 секунд: docker logs pd-api" }

Write-Host "`n--- /health ---"
Invoke-RestMethod http://localhost:8000/health | ConvertTo-Json

Write-Host "`n--- /predict (sample) ---"
$sample = @{
    LIMIT_BAL = 200000; AGE = 34; SEX = 1; EDUCATION = 2; MARRIAGE = 1
    PAY_0 = 2; PAY_2 = 2; PAY_3 = 1; PAY_4 = -1; PAY_5 = 0; PAY_6 = 0
    pay_max_status = 2; pay_min_status = -1; pay_mean_status = 0.67; pay_std_status = 1.2
    pay_num_late = 3; pay_num_severe = 2; pay_num_norecord = 0; pay_trend = 2
    pay_consec_late = 2; bill_mean = 50000; bill_max = 80000; bill_trend = 10000
    payamt_mean = 20000; payamt_max = 40000; pay_to_bill = 0.4
    age_group = 1; limit_bal_log = 12.2
}
Invoke-RestMethod -Method Post -Uri http://localhost:8000/predict `
    -ContentType "application/json" -Body ($sample | ConvertTo-Json) | ConvertTo-Json

Write-Host "`nAPI запущен: http://localhost:8000 (контейнер: pd-api)"
Write-Host "Остановка: docker rm -f pd-api"
