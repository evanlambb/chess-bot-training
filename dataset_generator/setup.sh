#!/bin/bash

# Linux/macOS setup script for Chess Dataset Generator

echo "========================================"
echo "Chess Dataset Generator - Setup"
echo "========================================"
echo

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "ERROR: Python 3 not found!"
    echo "Please install Python 3"
    exit 1
fi

echo "[1/3] Installing Python dependencies..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install dependencies"
    exit 1
fi

echo
echo "[2/3] Checking for Stockfish..."
if ! command -v stockfish &> /dev/null; then
    echo
    echo "WARNING: Stockfish not found in PATH"
    echo
    echo "Please install Stockfish:"
    echo "  macOS:  brew install stockfish"
    echo "  Linux:  sudo apt-get install stockfish"
    echo "          or sudo yum install stockfish"
    echo
    echo "You'll need to provide the path when running the generator."
    echo
else
    echo "Stockfish found: $(which stockfish)"
fi

echo
echo "[3/3] Setup complete!"
echo
echo "========================================"
echo "Next steps:"
echo "  1. Run: python3 quick_start.py"
echo "  2. Follow the interactive prompts"
echo "========================================"
echo

# Make scripts executable
chmod +x quick_start.py
chmod +x setup.sh

echo "Scripts made executable"

