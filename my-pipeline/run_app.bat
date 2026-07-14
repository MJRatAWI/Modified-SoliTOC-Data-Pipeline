@echo off
setlocal
cd /d "%~dp0"
set PYTHON_EXE=%~dp0.venv\Scripts\python.exe
if exist "%PYTHON_EXE%" (
  "%PYTHON_EXE%" app.py
) else (
  echo UV environment was not found at .venv\Scripts\python.exe
  echo Please run: uv sync
  exit /b 1
)
