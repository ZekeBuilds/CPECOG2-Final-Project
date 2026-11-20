# One-command setup for Windows. From the project folder:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Checks for Python 3.12, creates .venv, installs the pinned requirements, downloads the dataset
# and runs the tests. Safe to run again: finished steps are skipped or only re-checked.

# "Continue", because Windows PowerShell 5.1 can turn harmless stderr output from pip or
# TensorFlow into errors under "Stop". Every step checks $LASTEXITCODE instead.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot
$python312Url = "https://www.python.org/downloads/release/python-31210/"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"

function Write-Step($text) { Write-Host "`n== $text" -ForegroundColor Cyan }

function Stop-Setup($text) {
    Write-Host "`nSetup stopped: $text" -ForegroundColor Red
    exit 1
}

function Get-PythonVersion($exe, $extraArgs) {
    try {
        $v = & $exe @extraArgs -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($LASTEXITCODE -eq 0) { return "$v".Trim() }
    } catch {}
    return $null
}

# 1. Python 3.12 (TensorFlow 2.21 and the saved models were built with it)
Write-Step "Checking for Python 3.12"
$pyExe = $null
$pyArgs = @()
if ((Get-Command py -ErrorAction SilentlyContinue) -and ((Get-PythonVersion "py" @("-3.12")) -eq "3.12")) {
    $pyExe = "py"
    $pyArgs = @("-3.12")
} elseif ((Get-Command python -ErrorAction SilentlyContinue) -and ((Get-PythonVersion "python" @()) -eq "3.12")) {
    $pyExe = "python"
}
if (-not $pyExe) {
    $found = ""
    if (Get-Command python -ErrorAction SilentlyContinue) {
        $other = Get-PythonVersion "python" @()
        if ($other) { $found = " (found Python $other, but the project needs 3.12)" }
    }
    Stop-Setup ("Python 3.12 was not found$found. Install 'Windows installer (64-bit)' from " +
                "$python312Url (tick 'Add python.exe to PATH'), open a new terminal, and run this script again.")
}
Write-Host ("Found Python 3.12 (" + (@($pyExe) + $pyArgs -join " ") + ")")

# 2. Virtual environment
Write-Step "Virtual environment (.venv)"
if (Test-Path -LiteralPath $venvPython) {
    $venvVersion = Get-PythonVersion $venvPython @()
    if (-not $venvVersion) {
        Stop-Setup ".venv is broken (its Python does not start). Delete the .venv folder and run this script again."
    }
    if ($venvVersion -ne "3.12") {
        Stop-Setup (".venv uses Python $venvVersion, not 3.12. Delete the .venv folder and run this script again.")
    }
    Write-Host "Already exists, reusing it"
} else {
    & $pyExe @pyArgs -m venv .venv
    if ($LASTEXITCODE -ne 0) { Stop-Setup "could not create .venv" }
    Write-Host "Created .venv"
}

# 3. Requirements (pip skips packages that already match the pinned version)
Write-Step "Installing pinned requirements (the first time takes several minutes)"
& $venvPython -m pip install --quiet --disable-pip-version-check --upgrade pip
if ($LASTEXITCODE -ne 0) {
    Stop-Setup ".venv cannot run pip (it may be half-built). Delete the .venv folder and run this script again."
}
& $venvPython -m pip install --quiet --disable-pip-version-check -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Stop-Setup ("pip install -r requirements.txt failed (see the error above). If it mentions a long path, " +
                "move the project to a shorter folder such as C:\cpecog2 and run again.")
}
Write-Host "Requirements satisfied"

# 4. Dataset (skipped if already downloaded)
Write-Step "Dataset (UJIIndoorLoc)"
& $venvPython -m scripts.download_data
if ($LASTEXITCODE -ne 0) { Stop-Setup "dataset download failed. Check your internet connection and run again." }

# 5. Tests
Write-Step "Running the tests"
$env:TF_CPP_MIN_LOG_LEVEL = "2"
& $venvPython -m pytest -q tests -p no:warnings
if ($LASTEXITCODE -ne 0) { Stop-Setup "some tests failed (see above)" }

Write-Host "`nSetup complete. Start the demo with:" -ForegroundColor Green
Write-Host "  powershell -ExecutionPolicy Bypass -File run_demo.ps1"
