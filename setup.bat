@echo off
setlocal enabledelayedexpansion

:: 1. Change to the project root directory automatically
cd /d "%~dp0"

echo ====================================================
echo        Temporal-Multimodal-RAG Setup
echo ====================================================

set "REPAIR_MODE=0"
if /i "%1"=="--repair" set "REPAIR_MODE=1"
if /i "%1"=="repair" set "REPAIR_MODE=1"

:: 2. Determine Python 3.10 interpreter for virtual environment creation
set "PYTHON_EXE="

:: Check if virtual environment already exists and is healthy
if exist ".venv\Scripts\python.exe" (
    if "!REPAIR_MODE!"=="0" (
        echo [INFO] Existing virtual environment found at .venv\Scripts\python.exe.
        echo Checking Python version...
        ".venv\Scripts\python.exe" -c "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 10) else 1)" >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
            echo [OK] Existing virtual environment uses Python 3.10+.
            goto :install_deps
        ) else (
            echo [WARN] Existing virtual environment does not use Python 3.10+. Re-creating...
            set "REPAIR_MODE=1"
        )
    )
)

:: If repair requested, remove old .venv directory
if "!REPAIR_MODE!"=="1" (
    echo [INFO] Repairing virtual environment...
    if exist ".venv" (
        echo Removing old virtual environment...
        rmdir /s /q ".venv" >nul 2>&1
    )
)

:: Look for Python 3.10 on the system
echo Looking for Python 3.10 installation...

:: Try py -3.10 launcher
py -3.10 -V >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "SYS_PYTHON=py -3.10"
    goto :create_venv
)

:: Try python command
python -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 10) else 1)" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "SYS_PYTHON=python"
    goto :create_venv
)

:: Check default Windows installation directory
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "SYS_PYTHON=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :create_venv
)

:: Check Program Files
if exist "C:\Program Files\Python310\python.exe" (
    set "SYS_PYTHON=C:\Program Files\Python310\python.exe"
    goto :create_venv
)

echo.
echo ====================================================
echo [ERROR] Python 3.10 was not found on your system!
echo.
echo Please install Python 3.10 (64-bit) from python.org.
echo Ensure 'Add Python to PATH' and 'py launcher' are selected.
echo ====================================================
exit /b 1

:create_venv
echo [INFO] Creating Python virtual environment using: !SYS_PYTHON!
!SYS_PYTHON! -m venv .venv
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Failed to create virtual environment in .venv.
    exit /b 1
)
set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
echo [OK] Virtual environment created successfully.

:install_deps
:: 3. Install dependencies from requirements.txt
echo.
echo [INFO] Installing / verifying dependencies from requirements.txt...
"!PYTHON_EXE!" -m pip install -r requirements.txt
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Dependency installation failed. Check network or package errors above.
    exit /b 1
)
echo [OK] Dependencies successfully verified.

:: 4. Verify Dynamic FFmpeg resolution
echo.
echo [INFO] Checking FFmpeg dynamic discovery...
"!PYTHON_EXE!" -c "from processing.transcription import ensure_ffmpeg; ok, path = ensure_ffmpeg(); assert ok, 'FFmpeg discovery failed'; print('  Resolved FFmpeg binary:', path)"
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] FFmpeg dynamic discovery failed.
    exit /b 1
)
echo [OK] FFmpeg dynamic resolution verified.

:: 5. Verify core library imports
echo.
echo [INFO] Verifying core imports (Flask, Whisper, PyTorch, yt-dlp)...
"!PYTHON_EXE!" -c "import flask, werkzeug, whisper, torch, imageio_ffmpeg, youtube_transcript_api, yt_dlp, requests; print('  Core modules loaded successfully.')"
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Core module verification failed.
    exit /b 1
)
echo [OK] All core packages imported successfully.

:: 6. Run automated tests
echo.
echo [INFO] Running automated test suite...
"!PYTHON_EXE!" -m unittest discover -s tests
if !ERRORLEVEL! neq 0 (
    echo.
    echo [ERROR] Automated tests encountered failures. Review output above.
    exit /b 1
)
echo [OK] All automated tests passed.

:: 7. Ensure uploads folder exists
if not exist "uploads" mkdir uploads

echo.
echo ====================================================
echo           SETUP COMPLETED SUCCESSFULLY!
echo ====================================================
echo [OK] Python 3.10 virtual environment verified
echo [OK] Project dependencies installed
echo [OK] FFmpeg dynamic discovery verified
echo [OK] Core modules imported successfully
echo [OK] Automated test suite passed
echo.
echo You can now start the application anytime with:
echo     run.bat
echo ====================================================
exit /b 0
