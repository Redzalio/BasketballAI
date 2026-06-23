@echo off
cd /d "%~dp0"
title Install PyTorch for HoopTracker
echo ============================================================
echo   Installing PyTorch - the GPU build for your RTX 5070 Ti.
echo   This is a ONE-TIME download (~2.5 GB) and can take a few
echo   minutes. Leave this window open until you see DONE below.
echo ============================================================
echo.
set "PY=C:\Program Files\Python314\python.exe"
if not exist "%PY%" set "PY=py -3"
echo Using Python: %PY%
echo.
"%PY%" -m pip install --user torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
echo.
if errorlevel 1 (
  echo ************************************************************
  echo   INSTALL FAILED. Copy everything above and send to Claude.
  echo ************************************************************
) else (
  echo ============================================================
  echo   DONE - PyTorch is installed. Close this and run start.bat
  echo ============================================================
)
pause
