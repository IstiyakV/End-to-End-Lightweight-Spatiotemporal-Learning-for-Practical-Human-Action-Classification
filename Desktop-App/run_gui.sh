#!/bin/bash

echo "==================================================="
echo "🚀 HAR Control Center - Desktop GUI Launch Script"
echo "==================================================="
echo

# Check Python installation
if ! command -v python3 &> /dev/null; then
    echo "❌ ERROR: python3 could not be found. Please install Python 3.9+."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "env" ]; then
    echo "📦 Creating Virtual Environment [env]..."
    python3 -m venv env
    if [ $? -ne 0 ]; then
        echo "❌ ERROR: Failed to create virtual environment. Make sure python3-venv is installed."
        exit 1
    fi
fi

# Activate virtual environment
echo "⚡ Activating Virtual Environment..."
source env/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
python3 -m pip install --upgrade pip -q

# Install dependencies
echo "📥 Installing Pinned Dependencies [customtkinter, PyTorch, etc.]..."
echo "This may take a minute. Please wait..."
pip install -r requirements.txt -q
if [ $? -ne 0 ]; then
    echo "❌ ERROR: Dependency installation failed."
    exit 1
fi

# Launch the GUI app
echo "🎉 Launching HAR Control Center..."
python3 gui.py

if [ $? -ne 0 ]; then
    echo "⚠️ WARNING: Application exited with code $?."
fi
