@echo off
cd /d "%~dp0"
echo Starting HoopTracker...
start "" http://127.0.0.1:8791
python app.py
