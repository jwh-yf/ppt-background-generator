@echo off
setlocal
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8

echo ============================================
echo  pptbg - one click setup and run
echo ============================================
echo.

echo [1/4] Checking Python ...
where python >nul 2>nul
if errorlevel 1 (
    echo   ERROR: python not found.
    echo   Install Python 3.10-3.12 from https://www.python.org/downloads/
    echo   Remember to check "Add python.exe to PATH" during install.
    echo.
    pause
    exit /b 1
)
python --version
echo.

echo [2/4] Preparing virtual environment .venv ...
if not exist ".venv\Scripts\python.exe" (
    python -m venv .venv
    if errorlevel 1 (
        echo   ERROR: failed to create .venv
        pause
        exit /b 1
    )
    echo   created.
) else (
    echo   already exists, skipped.
)
echo.

echo [3/4] Installing dependencies (first run: 5-10 min) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>nul
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo.
    echo   ERROR: pip install failed. Scroll up for the real error.
    echo   Common fix: run this again, or delete .venv and rerun.
    pause
    exit /b 1
)
echo.

set HF_HOME=%cd%\tools\hf
if not exist "%HF_HOME%\stable-diffusion-v1-5\model_index.json" (
    echo   NOTE: base model weights not found.
    echo         The app will still run, using the procedural fallback
    echo         ^(background drawn by rules, no LoRA^). All UI and export
    echo         features work the same.
    echo         To get model quality: tools\fetch_model.py sd15 ^(3.1 GB^)
    echo.
)

echo [4/4] Starting server at http://127.0.0.1:8763 ...
echo       Press Ctrl+C in this window to stop.
echo.
start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8763/"
".venv\Scripts\python.exe" app\server.py --port 8763

echo.
echo Server stopped.
pause
