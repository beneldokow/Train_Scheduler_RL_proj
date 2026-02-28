#!/bin/bash

# Train Scheduler RL - Unified Entry Point Script
# 
# Responsibilities:
# 1. Environment Management: Automatically creates and activates a Python virtual environment.
# 2. Dependency Sync: Ensures 'pip' and project requirements are up-to-date on every run.
# 3. Filesystem Resilience: Detects restricted environments (like Google Drive) and redirects 
#    the virtual environment to a safe local path (~/venv_train_scheduler_rl) to avoid symlink errors.
# 4. Interactive Helpers: Simplifies complex flags (e.g., provides a list for --reuse).
# 5. Main Execution: Launches 'src/train.py' with processed arguments.

# Configuration
VENV_PATH_FILE=".venv_path" # Persistent file to remember the external venv location
VENV_DIR=""

# 0. Pre-flight Check: Verify Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is not installed. Please install it first (e.g., sudo apt install python3)."
    exit 1
fi

# Function: is_venv_valid
# Checks if a directory contains a functional Python virtual environment.
is_venv_valid() {
    local dir="$1"
    if [ -d "$dir" ] && [ -f "$dir/bin/activate" ] && [ -f "$dir/bin/python3" ]; then
        # Google Drive does not support symlinks, which breaks venv functionality.
        if [[ "$dir" == /mnt/chromeos/GoogleDrive/* ]]; then
            return 1 
        fi
        return 0
    fi
    return 1
}

# 1. Resolve Environment: Check saved path first
if [ -f "$VENV_PATH_FILE" ]; then
    VENV_DIR=$(cat "$VENV_PATH_FILE")
    if ! is_venv_valid "$VENV_DIR"; then
        echo "Warning: Saved virtual environment at $VENV_DIR is invalid or on an unsupported filesystem."
        VENV_DIR=""
    fi
fi

# 2. Resolve Environment: Check for local 'venv' folder
if [ -z "$VENV_DIR" ] && is_venv_valid "venv"; then
    VENV_DIR="venv"
fi

# 3. Environment Setup Flow
if [ -z "$VENV_DIR" ]; then
    echo "No valid virtual environment found."

    CURRENT_DIR=$(pwd)
    if [[ "$CURRENT_DIR" == /mnt/chromeos/GoogleDrive/* ]]; then
        echo "Detected Google Drive. Local virtual environments are not supported here due to symlink restrictions."
        DEFAULT_VENV="$HOME/venv_train_scheduler_rl"
    else
        DEFAULT_VENV="$CURRENT_DIR/venv"
    fi

    echo "Would you like to set up a new virtual environment at: $DEFAULT_VENV? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Creating virtual environment..."
        mkdir -p "$(dirname "$DEFAULT_VENV")"
        python3 -m venv "$DEFAULT_VENV"
        if [ $? -eq 0 ]; then
            VENV_DIR="$DEFAULT_VENV"
            echo "$VENV_DIR" > "$VENV_PATH_FILE"
            echo "Virtual environment created at $VENV_DIR"
        else
            echo "------------------------------------------------"
            echo "Failed to create virtual environment."
            echo "This usually means the 'python3-venv' package is missing."
            echo "Fix: sudo apt update && sudo apt install python3-venv python3-pip"
            echo "------------------------------------------------"
            exit 1
        fi
    else
        echo "Please provide the path to your existing virtual environment (e.g. ~/my_venv):"
        read -r custom_path
        eval custom_path="$custom_path" # Expand ~
        if is_venv_valid "$custom_path"; then
            VENV_DIR="$custom_path"
            echo "$VENV_DIR" > "$VENV_PATH_FILE"
        else
            echo "Error: The path '$custom_path' is not a valid virtual environment."
            exit 1
        fi
    fi
fi

# 4. Activation & Verification
source "$VENV_DIR/bin/activate"
echo "Verifying dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# 5. Argument Preprocessing
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --reuse)
      if [ -n "$2" ] && [[ ! "$2" == --* ]]; then
        INSTANCE_NAME="$2"
        shift 2
      else
        echo "Available reused instances:"
        if [ -d "rddl/instances" ]; then
          ls -1 rddl/instances/ | sed 's/\.rddl$//'
        else
          echo "(None found in rddl/instances/)"
        fi
        echo "Enter instance name:"
        read -r INSTANCE_NAME
        shift
      fi
      ARGS+=("--instance_path" "rddl/instances/${INSTANCE_NAME}.rddl")
      ;;
    --instance)
      ARGS+=("--instance_path" "rddl/instances/$2.rddl")
      shift 2
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

# 6. Main Execution
python3 src/train.py "${ARGS[@]}"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Training Complete!"
    echo "Interactive Dashboard: output/training_dashboard.html"
    echo "------------------------------------------------"
fi
