#!/bin/bash
# TIGER Setup Script - Creates venv and installs dependencies

set -e  # Exit on error

echo "=========================================="
echo "Setting up TIGER environment..."
echo "=========================================="
echo ""

# Check if Python 3 is available
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: python3 not found. Please install Python 3.8 or later."
    exit 1
fi

# Display Python version
PYTHON_VERSION=$(python3 --version)
echo "✓ Found $PYTHON_VERSION"
echo ""

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv tiger_env
echo "✓ Virtual environment created at ./tiger_env"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source tiger_env/bin/activate
echo "✓ Virtual environment activated"
echo ""

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip --quiet
echo "✓ pip upgraded"
echo ""

# Install core dependencies
echo "Installing core dependencies..."
pip install -r requirements.txt --quiet
echo "✓ Core dependencies installed"
echo ""

# Install optional dependencies (non-blocking)
if [ -f requirements-optional.txt ]; then
    echo "Installing optional dependencies..."
    if pip install -r requirements-optional.txt --quiet 2>/dev/null; then
        echo "✓ Optional dependencies installed"
    else
        echo "⚠ Optional dependencies skipped (not critical)"
    fi
    echo ""
fi

# Install TIGER in development mode
echo "Installing TIGER in development mode..."
pip install -e . --quiet
echo "✓ TIGER installed"
echo ""

# Create example output directories
echo "Creating example output directories..."
mkdir -p examples/1D examples/2D examples/3D
echo "✓ Output directories created"
echo ""

echo "=========================================="
echo "Setup complete! 🎉"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  source tiger_env/bin/activate"
echo ""
echo "To run examples:"
echo "  cd examples"
echo "  python 1d_plot.py"
echo ""
echo "To deactivate when done:"
echo "  deactivate"
echo "=========================================="
