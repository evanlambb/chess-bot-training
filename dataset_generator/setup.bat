@echo off
REM Windows setup script for Chess Dataset Generator

echo ========================================
echo Chess Dataset Generator - Setup
echo ========================================
echo.

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python not found!
    echo Please install Python from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)

echo.
echo [2/3] Checking for Stockfish...
where stockfish >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo WARNING: Stockfish not found in PATH
    echo.
    echo Please download Stockfish from:
    echo   https://stockfishchess.org/download/
    echo.
    echo Extract it and remember the path to stockfish.exe
    echo You'll need to provide this path when running the generator.
    echo.
) else (
    echo Stockfish found in PATH!
)

echo.
echo [3/3] Setup complete!
echo.
echo ========================================
echo Next steps:
echo   1. Run: python quick_start.py
echo   2. Follow the interactive prompts
echo ========================================
echo.
pause

