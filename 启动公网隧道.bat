@echo off
setlocal

set "ROOT=%~dp0"
set "LOG=%ROOT%tunnel.log"

set "PYTHON=%ROOT%scripts\.venv\Scripts\python.exe"
if exist "%PYTHON%" goto python_ready
where python >nul 2>&1
if errorlevel 1 goto missing_python
set "PYTHON=python"

:python_ready
set "CLOUDFLARED=cloudflared"
where cloudflared >nul 2>&1
if not errorlevel 1 goto cloudflared_ready
set "CLOUDFLARED=C:\Program Files (x86)\cloudflared\cloudflared.exe"
if not exist "%CLOUDFLARED%" goto missing_cloudflared

:cloudflared_ready
cd /d "%ROOT%"

powershell -NoProfile -Command "try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if not errorlevel 1 goto api_ready

echo Starting FinRAG API...
start "FinRAG API" /min "%PYTHON%" "scripts\api_server.py"
echo Waiting for FinRAG API...
powershell -NoProfile -Command "for($i=0;$i -lt 120;$i++){ try { Invoke-WebRequest -Uri 'http://127.0.0.1:8000/api/health' -UseBasicParsing -TimeoutSec 1 | Out-Null; exit 0 } catch { Start-Sleep -Milliseconds 500 } }; exit 1" >nul 2>&1
if errorlevel 1 goto api_failed

:api_ready
echo FinRAG API is ready.

powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'cloudflared.exe' -and $_.CommandLine -like '*tunnel*--url*http://127.0.0.1:8000*' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force }" >nul 2>&1
if exist "%LOG%" del "%LOG%" >nul 2>&1
echo Starting public tunnel...
start "FinRAG Tunnel" /min "%CLOUDFLARED%" tunnel --url http://127.0.0.1:8000 --no-autoupdate --logfile "%LOG%"

set "URL="
for /l %%i in (1,1,120) do (
  if defined URL goto url_ready
  for /f "usebackq delims=" %%u in (`powershell -NoProfile -Command "if(Test-Path $env:LOG){(Select-String -LiteralPath $env:LOG -Pattern 'https://[^\s]+trycloudflare\.com' | Select-Object -First 1).Matches.Value}"`) do set "URL=%%u"
  if not defined URL timeout /t 1 >nul
)

:url_ready
if not defined URL goto tunnel_failed

echo.
echo Public URL: %URL%
start "" "%URL%"
echo.
echo Keep the FinRAG API and FinRAG Tunnel windows open while using this link.
echo Close both of those windows when you want to stop the public tunnel.
pause
exit /b 0

:missing_python
echo Python not found: %PYTHON%
pause
exit /b 1

:missing_cloudflared
echo cloudflared not found: %CLOUDFLARED%
pause
exit /b 1

:api_failed
echo FinRAG API failed to start.
pause
exit /b 1

:tunnel_failed
echo Failed to get the public URL. Check tunnel.log for details.
pause
exit /b 1
