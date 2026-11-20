# Start the Streamlit demo with the project's .venv. Run setup.ps1 once first.
#   powershell -ExecutionPolicy Bypass -File run_demo.ps1
# Then open http://localhost:8501. Stop it with Ctrl+C.

Set-Location -LiteralPath $PSScriptRoot
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $venvPython)) {
    Write-Host "No .venv found. Run setup first:  powershell -ExecutionPolicy Bypass -File setup.ps1" -ForegroundColor Red
    exit 1
}
$env:TF_CPP_MIN_LOG_LEVEL = "2"
Write-Host "Starting the demo. Open http://localhost:8501 (the first load takes about 30 seconds)." -ForegroundColor Green
Write-Host "If it says port 8501 is in use, a demo is already running: close that window first."
# Fixed port, so the URL above is always right: a busy port fails instead of moving to 8502.
& $venvPython -m streamlit run app.py --server.port 8501
exit $LASTEXITCODE
