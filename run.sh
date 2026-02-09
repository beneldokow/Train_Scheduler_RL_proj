#!/bin/bash

VENV_PATH_FILE=".venv_path"
VENV_DIR=""

# Function to check if a venv is actually valid and functional
is_venv_valid() {
    local dir="$1"
    if [ -d "$dir" ] && [ -f "$dir/bin/activate" ] && [ -f "$dir/bin/python3" ]; then
        # Check if it's on a restricted filesystem (Google Drive)
        if [[ "$dir" == /mnt/chromeos/GoogleDrive/* ]]; then
            return 1 # Invalid because it's on Drive (symlinks will fail)
        fi
        return 0
    fi
    return 1
}

# 1. Check if we already have a path saved
if [ -f "$VENV_PATH_FILE" ]; then
    VENV_DIR=$(cat "$VENV_PATH_FILE")
    if ! is_venv_valid "$VENV_DIR"; then
        echo "Warning: Saved virtual environment at $VENV_DIR is invalid or on an unsupported filesystem."
        VENV_DIR=""
    fi
fi

# 2. Check if local venv exists and use it if valid
if [ -z "$VENV_DIR" ] && is_venv_valid "venv"; then
    VENV_DIR="venv"
fi

# 3. If no venv found or the saved one is invalid, help the user
if [ -z "$VENV_DIR" ]; then
    echo "No valid virtual environment found."
    
    # Detect Google Drive / Restricted filesystem
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
        # Ensure parent directory exists
        mkdir -p "$(dirname "$DEFAULT_VENV")"
        python3 -m venv "$DEFAULT_VENV"
        if [ $? -eq 0 ]; then
            VENV_DIR="$DEFAULT_VENV"
            echo "$VENV_DIR" > "$VENV_PATH_FILE"
            echo "Virtual environment created at $VENV_DIR"
        else
            echo "Failed to create virtual environment. Please create one manually and put its path in $VENV_PATH_FILE"
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
            echo "Error: The path '$custom_path' is not a valid virtual environment or is on an unsupported filesystem."
            exit 1
        fi
    fi
fi

# Activate
source "$VENV_DIR/bin/activate"

# 4. ROBUSTNESS: Check and install missing dependencies
# This ensures that even if the venv exists, it has everything it needs
echo "Verifying dependencies..."
pip install -q -r requirements.txt

# Argument preprocessing for convenience
ARGS=()
while [[ $# -gt 0 ]]; do
  case $1 in
    --reuse)
      if [ -n "$2" ] && [[ ! "$2" == --* ]]; then
        INSTANCE_NAME="$2"
        shift 2
      else
        echo "Available reused instances:"
        ls -1 rddl/instances/ | sed 's/\.rddl$//'
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

# Run the training script with processed arguments
python3 src/train.py "${ARGS[@]}"

if [ $? -eq 0 ]; then
    echo "------------------------------------------------"
    echo "Training Complete!"
    echo "Interactive Dashboard: output/training_dashboard.html"
    echo "------------------------------------------------"
fi
