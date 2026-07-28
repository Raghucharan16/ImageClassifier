@echo off
REM Launch the Offline Image Classifier.
REM Prefers the standalone .exe (no Python needed); falls back to source.
cd /d "%~dp0"
if exist "dist\run_offline.exe" (
    "dist\run_offline.exe"
) else if exist "run_offline.exe" (
    "run_offline.exe"
) else (
    python run_offline.py
)
pause
