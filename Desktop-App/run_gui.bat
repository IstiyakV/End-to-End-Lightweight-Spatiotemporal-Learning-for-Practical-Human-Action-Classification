@echo off
echo ===================================================
echo 🚀 HAR Control Center - Desktop GUI Launch Script
echo ===================================================
echo.

:: Check Python installation
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ ERROR: Python 3.9+ is not installed or not in system PATH.
    echo Please install Python and try again.
    pause
    exit /b 1
)

:: Create virtual environment if it doesn't exist
if not exist "env" (
    echo 📦 Creating Virtual Environment (env)...
    python -m venv env
    if %errorlevel% neq 0 (
        echo ❌ ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate virtual environment
echo ⚡ Activating Virtual Environment...
call env\Scripts\activate

:: Upgrade pip
echo 🔄 Upgrading pip...
python -m pip install --upgrade pip -q

:: Install dependencies
echo 📥 Installing Pinned Dependencies (customtkinter, PyTorch, etc.)...
echo This may take a minute. Please wait...
pip install -r requirements.txt -q
if %errorlevel% neq 0 (
    echo ❌ ERROR: Dependency installation failed.
    pause
    exit /b 1
)

:: Launch the GUI app
echo 🎉 Launching HAR Control Center...
python gui.py
if %errorlevel% neq 0 (
    echo ⚠️ WARNING: Application exited with code %errorlevel%.
)

pause
