@echo off
setlocal

:: 1. Change to the project root directory automatically
cd /d "%~dp0"
set "PYTHONUNBUFFERED=1"

:: 2. Check that the required Python environment exists at .venv\Scripts\python.exe
if not exist ".venv\Scripts\python.exe" (
    echo ====================================================
    echo [ERROR] Python environment not found!
    echo   Target: %~dp0.venv\Scripts\python.exe
    echo.
    echo Please run 'setup.bat' first to configure Python 3.10
    echo and install required dependencies.
    echo ====================================================
    if not "%1"=="--no-pause" pause
    exit /b 1
)

:: 3. Run the startup coordinator using the virtual environment
".venv\Scripts\python.exe" run.py %*
set "EXIT_CODE=%ERRORLEVEL%"

if %EXIT_CODE% neq 0 (
    echo.
    echo [ERROR] Application exited with code %EXIT_CODE%.
    if not "%1"=="--no-pause" pause
)

exit /b %EXIT_CODE%
