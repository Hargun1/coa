@echo off
setlocal
cd /d "%~dp0"
echo Adaptive Memory Lab
echo Open http://127.0.0.1:8765 in your browser.
echo Press Ctrl+C to stop the server.
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -m adaptive_memory serve --port 8765
) else (
  python -m adaptive_memory serve --port 8765
)
if errorlevel 1 pause
