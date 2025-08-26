#!/bin/bash

# Setup and activate virtual environment for Coherify

echo "🚀 Coherify Virtual Environment Setup"
echo "======================================"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Verify activation
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✅ Virtual environment activated: $VIRTUAL_ENV"
else
    echo "❌ Failed to activate virtual environment"
    exit 1
fi

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet

# Install requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt --quiet

# Install coherify in editable mode
echo "📦 Installing Coherify in development mode..."
pip install -e . --quiet

# Verify installation
echo ""
echo "🔍 Verifying installation..."
python scripts/verify_environment.py

echo ""
echo "======================================"
echo "✅ Setup complete!"
echo ""
echo "To activate the virtual environment in the future, run:"
echo "  source venv/bin/activate"
echo ""
echo "To test the pipeline, run:"
echo "  python examples/run_full_pipeline_comparison.py --model gpt4-mini --samples 5"