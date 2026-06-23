@echo off
cd /d "%~dp0"
title HoopTracker Server
echo ==============================================
echo   Starting HoopTracker...
echo   Loading models - first launch can take
echo   10-20 seconds. The browser opens by itself
echo   the moment the server is ready. Keep this
echo   window open while you use the app.
echo ==============================================
echo.

REM Use the real Python, NOT the Microsoft Store stub.
set "PY=C:\Program Files\Python314\python.exe"
if not exist "%PY%" set "PY=py -3"

REM run.py fixes its own import path AND opens the browser from inside Python
REM (no PowerShell helper, so nothing for antivirus to flag). Runs in THIS
REM window so any error stays visible.
"%PY%" run.py

echo.
echo HoopTracker stopped. Press any key to close.
pause >nul
