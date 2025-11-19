# One-click launcher for Vietnamese News NLP Portal
# - Sets environment variables
# - Starts FastAPI backend with auto-crawl
# - Opens frontend in default browser

$ErrorActionPreference = "Stop"

# Determine project root based on script location
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

Write-Host "[run_app] Project root: $ProjectRoot" -ForegroundColor Cyan

# ==== Environment variables (edit if needed) ====
$env:DATABASE_URL = "postgresql+psycopg2://postgres:1234@localhost:6699/news_nlp"
$env:AUTO_CRAWL_ENABLED = "true"
$env:AUTO_CRAWL_INTERVAL_MINUTES = "10"   # 10 phút / lần
$env:AUTO_CRAWL_LIMIT_PER_FEED = "50"     # tối đa 50 bài / nguồn mỗi lần crawl

Write-Host "[run_app] DATABASE_URL=$($env:DATABASE_URL)" -ForegroundColor Yellow
Write-Host "[run_app] Auto-crawl: enabled=$($env:AUTO_CRAWL_ENABLED), interval=$($env:AUTO_CRAWL_INTERVAL_MINUTES) minutes, limit_per_feed=$($env:AUTO_CRAWL_LIMIT_PER_FEED)" -ForegroundColor Yellow

# ==== Start backend in new PowerShell window ====
$backendCmd = "cd `"$ProjectRoot`"; python -m uvicorn app.main:app --reload --port 8000"
Write-Host "[run_app] Starting backend: $backendCmd" -ForegroundColor Green
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd

Start-Sleep -Seconds 10  # đợi backend khởi động (nạp model PhoBERT, kết nối DB)

# ==== Open frontend in default browser ====
$frontendPath = Join-Path $ProjectRoot "frontend\index.html"
if (Test-Path $frontendPath) {
    Write-Host "[run_app] Opening frontend: $frontendPath" -ForegroundColor Green
    Start-Process $frontendPath
} else {
    Write-Warning "[run_app] Frontend file not found: $frontendPath"
}

Write-Host "[run_app] Done. Backend window is running separately." -ForegroundColor Cyan
