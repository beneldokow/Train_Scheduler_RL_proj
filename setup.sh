#!/bin/bash

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv

# Activate and install
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Generate initial instance
echo "Generating initial RDDL instance..."
python3 -c "import os, sys; sys.path.insert(0, os.getcwd()); from src.generator import generate_instance; content = generate_instance(3, 4, 50); os.makedirs('rddl', exist_ok=True); open('rddl/instance.rddl', 'w').write(content)"

echo "Setup complete! Run './run.sh' to start training."
