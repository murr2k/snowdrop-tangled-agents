@echo off
REM Setup script for Windows
REM Usage: setup.bat [--dev] [--no-browsers]

echo ==========================================
echo   Snowdrop Tangled Agents Setup
echo ==========================================

REM Check for Python
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found. Please install Python 3.11 or later.
    exit /b 1
)

REM Run the Python setup script
python setup_env.py %*
