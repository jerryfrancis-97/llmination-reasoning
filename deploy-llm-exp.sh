#!/bin/bash

# --- Step 1: Prepare the Environment ---
echo "Setting up the environment..."

# Download and install Miniconda (if not already installed)
if ! command -v conda &> /dev/null; then
  echo "Miniconda not found. Downloading and installing..."
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda3
  source $HOME/miniconda3/etc/profile.d/conda.sh
  conda init bash -q # Quietly initialize bash
  source ~/.bashrc
  rm -f miniconda.sh
  echo "Miniconda installed."
else
  echo "Miniconda is already installed."
fi

# Update system packages and install necessary tools
echo "Updating system and installing build tools..."
sudo apt update &>/dev/null # Keep output clean
sudo apt install -y build-essential cmake &>/dev/null
echo "System tools are ready."

# --- Step 2: Get the Code ---
REPO_URL="https://${GITHUB_TOKEN}@github.com/jerryfrancis-97/llmination-reasoning.git"
BRANCH_NAME="jerome-infer-dev"
REPO_DIR="llmination-reasoning"

echo "Getting the code from GitHub (branch: '$BRANCH_NAME')..."
if [ -d "$REPO_DIR" ]; then
  echo "Code directory '$REPO_DIR' already exists."
  cd "$REPO_DIR"
  git checkout "$BRANCH_NAME" &>/dev/null
  git pull origin "$BRANCH_NAME" &>/dev/null
  echo "Code updated."
else
  git clone --recurse-submodules -b "$BRANCH_NAME" "$REPO_URL" "$REPO_DIR"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to get the code. Make sure your GitHub token is set (export GITHUB_TOKEN='...') and is correct."
    exit 1
  fi
fi
cd "$REPO_DIR"
echo "Code is ready in '$REPO_DIR'."

# --- Step 3: Set Up the Software Environment ---
ENV_FILE="steering-thinking-models/environment-mini.yaml"
ENV_NAME="simple_env"

echo "Setting up the software environment using Conda ('$ENV_NAME')..."
if conda env list | grep -q "$ENV_NAME"; then
  echo "Conda environment '$ENV_NAME' already exists."
  conda activate "$ENV_NAME"
  echo "Environment activated."
else
  conda env create -f "$ENV_FILE" --name "$ENV_NAME"
  if [ $? -ne 0 ]; then
    echo "Error: Failed to create the Conda environment. Check the file '$ENV_FILE'."
    exit 1
  fi
  conda activate "$ENV_NAME"
  echo "Conda environment '$ENV_NAME' created and activated."
fi

# --- Step 4: Run the Code ---
EXEC_DIR="steering-thinking-models/compare-base-reasoning"
RUN_SCRIPT="./run.sh"

echo "Moving to the code execution directory: '$EXEC_DIR'..."
cd "$EXEC_DIR"

if [ $? -ne 0 ]; then
  echo "Error: Could not go to '$EXEC_DIR'."
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

echo "All done!"



