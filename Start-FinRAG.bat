@echo off
setlocal
title FinRAG

set "PYTHON=C:\Users\25328\AppData\Local\Programs\Python\Python312\python.exe"
set "ROOT=%~dp0"

cd /d "%ROOT%"

if not exist "%PYTHON%" (
  echo Python not found: C:\Users\25328\AppData\Local\Programs\Python\Python312\python.exe
  pause
  exit /b 1
)

echo Starting FinRAG...
echo Close this window to stop the service.

start "" /min powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%scripts\open_browser.ps1"

"%PYTHON%" "scripts\api_server.py"

echo.
echo FinRAG service stopped.
pause
