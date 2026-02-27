#!/bin/bash

# YOLO11n-Project: Automated Run Script

set -e

# 1. Activate or create Python virtual environment (.venv)
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

# 2. Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# 3. Install requirements
echo "Installing requirements from requirements.txt..."
pip install -r requirements.txt

# 4. Run the main project script
echo "Running main.py..."
python src/main.py

echo "All done!"
