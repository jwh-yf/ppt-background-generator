@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

set "PY="
set "PYDESC="

echo ============================================
echo  pptbg - one click setup and run
echo ============================================
echo.

echo [1/5] Looking for Python 3.10-3.12 ...
rem torch has no wheels for 3.13+, so picking the interpreter matters.
for %%V in (3.12 3.11 3.10) do (
  if not defined PY (
    py -%%V -c "import sys" >nul 2>nul
    if not errorlevel 1 (
      set "PY=py -%%V"
      set "PYDESC=py -%%V"
    )
  )
)
if not defined PY (
  python -c "import sys" >nul 2>nul
  if not errorlevel 1 (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
    if not errorlevel 1 (
      set "PY=python"
      set "PYDESC=python"
    )
  )
)
if not defined PY (
  echo.
  echo   ERROR: no Python 3.10-3.12 found.
  echo.
  echo   Your default python may be 3.13 or newer - torch has no wheels for it.
  echo   Install Python 3.12 with one of these:
  echo     1) winget install --id Python.Python.3.12 -e
  echo     2) https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe
  echo     3) https://mirrors.aliyun.com/python-release/windows/python-3.12.10-amd64.exe
  echo   python.org is often unreachable in mainland China.
  echo   During install, check "Add python.exe to PATH".
  echo   Then run this file again.
  echo.
  pause
  exit /b 1
)
echo   using: !PYDESC!

echo.
echo [2/5] Preparing virtual environment .venv ...
if exist ".venv\Scripts\python.exe" (
  .venv\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,12) else 1)" >nul 2>nul
  if errorlevel 1 (
    echo   .venv was built with an unsupported Python, recreating ...
    rmdir /s /q .venv
  )
)
if not exist ".venv\Scripts\python.exe" (
  !PY! -m venv .venv
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
echo [3/5] Upgrading pip ...
".venv\Scripts\python.exe" -m pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple >nul 2>nul

echo.
echo [4/5] Installing dependencies ^(first run: 5-15 min, torch is ~1 GB^) ...
echo   If this line fails, see "troubleshooting" in VERIFY.md.
".venv\Scripts\python.exe" -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
  echo.
  echo   ERROR: pip install failed. Scroll up for the real error.
  echo   Common causes:
  echo     - Python version is not 3.10-3.12
  echo     - no disk space ^(need about 5 GB^)
  echo     - network blocked; try again later
  echo.
  pause
  exit /b 1
)

echo.
echo [5/5] Starting server on http://127.0.0.1:8763 ...
if not exist "tools\hf\stable-diffusion-v1-5\model_index.json" (
  echo   NOTE: SD1.5 weights not found. The app still runs, using the
  echo         procedural fallback - no model, but backgrounds still work.
  echo         To get model output: .venv\Scripts\python.exe tools\fetch_model.py sd15
  echo.
)

start "" cmd /c "timeout /t 4 >nul & start http://127.0.0.1:8763/"
".venv\Scripts\python.exe" app\server.py --port 8763

echo.
echo Server stopped.
pause
