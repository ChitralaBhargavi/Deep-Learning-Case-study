@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title RetinaAI launcher

where python >nul 2>nul || ( echo [ERROR] Python not found. Install Python 3.10-3.13 and tick "Add to PATH". & pause & exit /b 1 )
where npm >nul 2>nul || ( echo [ERROR] Node.js not found. Install Node.js LTS from nodejs.org. & pause & exit /b 1 )

if not exist "venv\Scripts\python.exe" python -m venv venv
call "venv\Scripts\activate.bat"
if not exist "venv\.deps_installed" (
    python -m pip install --upgrade pip
    python -m pip install -r backend\requirements.txt || ( echo [ERROR] pip install failed & pause & exit /b 1 )
    echo installed> "venv\.deps_installed"
)
if not exist "frontend\node_modules" ( pushd frontend & call npm install --no-audit --no-fund & popd )

start "RetinaAI Backend" /d "%~dp0backend" cmd /k "call ..\venv\Scripts\activate.bat && python -m uvicorn main:app --reload"
start "RetinaAI Website" /d "%~dp0frontend" cmd /k "npm run dev"
timeout /t 12 /nobreak >nul
start "" "http://localhost:5173"
