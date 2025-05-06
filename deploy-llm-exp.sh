#!/bin/bash

# --- Step 1: Prepare the Environment ---
echo "Setting up the environment..."

# Update system packages and install necessary tools
echo "Updating system and installing build tools..."
sudo apt update &>/dev/null # Keep output clean
sudo apt install -y build-essential cmake python3 python3-venv python3-pip &>/dev/null
echo "System tools are ready."

# --- Step 2: Get the Code ---
REPO_URL="https://${GITHUB_TOKEN}@github.com/jerryfrancis-97/llmination-reasoning.git"
BRANCH_NAME="jerome-inference-dev"
REPO_DIR="llmination-reasoning"

echo "Getting the code from GitHub (branch: '$BRANCH_NAME')..."
if [ -d "$REPO_DIR" ]; then
  echo "Code directory '$REPO_DIR' already exists."
  cd "$REPO_DIR"
  git checkout "$BRANCH_NAME" &>/dev/null
  git pull origin "$BRANCH_NAME" &>/dev/null
  echo "Code updated."
else
  git clone -b "$BRANCH_NAME" "$REPO_URL" "$REPO_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to get the code. Make sure your GitHub token is set (export GITHUB_TOKEN='...') and is correct."
    exit 1
  fi
fi
cd "$REPO_DIR"
echo "Code is ready in '$REPO_DIR'."

# --- Step 3: Set Up the Software Environment ---
ENV_DIR="venv"
REQUIREMENTS_FILE="steering-thinking-models-main/requirements.txt"

echo "Setting up the software environment using Python virtual environment..."
if [ -d "$ENV_DIR" ]; then
  echo "Virtual environment '$ENV_DIR' already exists."
else
  python3 -m venv "$ENV_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create the virtual environment."
    exit 1
  fi
  echo "Virtual environment '$ENV_DIR' created."
fi

# Activate the virtual environment
source "$ENV_DIR/bin/activate"

# Install dependencies
if [ -f "$REQUIREMENTS_FILE" ]; then
  pip install --upgrade pip &>/dev/null
  pip install -r "$REQUIREMENTS_FILE"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to install dependencies from '$REQUIREMENTS_FILE'."
    deactivate
    exit 1
  fi
  echo "Dependencies installed."
else
  echo "Error: Requirements file '$REQUIREMENTS_FILE' not found."
  deactivate
  exit 1
fi

# --- Step 4: Run the Code ---
EXEC_DIR="steering-thinking-models-main/compare-base-reasoning"
RUN_SCRIPT="./run.sh"

echo "Moving to the code execution directory: '$EXEC_DIR'..."
cd "$EXEC_DIR"

if [ $? -ne 0 ]; then
  echo "Error: Could not go to '$EXEC_DIR'."
  deactivate
  exit 1
fi

echo "Changing permissions for the run script..."
chmod +x "$RUN_SCRIPT"
if [ $? -ne 0 ]; then
  echo "Error: Could not change permissions for '$RUN_SCRIPT'."
  deactivate
  exit 1
fi

echo "Running the main script: '$RUN_SCRIPT'..."

if [ -f "$RUN_SCRIPT" ]; then
  bash "$RUN_SCRIPT"
  if [ $? -ne 0 ]; then
    echo "Error: The script '$RUN_SCRIPT' had a problem."
  fi
else
  echo "Error: The script '$RUN_SCRIPT' was not found."
fi

# Deactivate the virtual environment
deactivate

echo "All done!"