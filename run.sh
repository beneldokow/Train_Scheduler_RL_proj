#!/bin/bash

# Define virtual environment path
VENV_DIR="/home/beneldokow/venv_train_rl"

# Check if venv exists
if [ ! -d "$VENV_DIR" ]; then
    echo "Virtual environment not found at $VENV_DIR!"
    echo "Please run ./setup.sh first."
    exit 1
fi

# Activate virtual environment
source "$VENV_DIR/bin/activate"

# Run the training script with all passed arguments
python3 src/train.py "$@"
