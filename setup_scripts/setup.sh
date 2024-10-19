#!/bin/bash

# Determine the directory of the current script
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Set the grandparent directory (two levels up from the current script)
GRANDPARENT_DIR="$(dirname "$SCRIPT_DIR")"

# Update PYTHONPATH to include the grandparent directory
export PYTHONPATH="$GRANDPARENT_DIR:$PYTHONPATH"

# Debugging: print the grandparent directory and PYTHONPATH
echo "SCRIPT_DIR: $SCRIPT_DIR"
echo "GRANDPARENT_DIR: $GRANDPARENT_DIR"
echo "PYTHONPATH: $PYTHONPATH"

# Fetch the environment name from config_shell.py inside the CONFIG folder using Python
ENV_NAME=$(python -c "import CONFIG.config_shell as config; print(config.ENV_NAME)" 2> /dev/null)

# Check if the environment name is retrieved
if [ -z "$ENV_NAME" ]
then
    echo "Environment name could not be retrieved from CONFIG/config_shell.py."
    exit 1
fi

# Ensure conda is available
if ! command -v conda &> /dev/null
then
    echo "Conda could not be found. Please install Miniconda or Anaconda and try again."
    exit 1
fi

# Check if the conda environment already exists
if conda info --envs | grep -q "^$ENV_NAME\s"; then
    echo "Environment '$ENV_NAME' already exists. Activating it."
else
    # Create the conda environment from the environment.yml file if not found
    echo "Creating environment '$ENV_NAME' from environment.yml."
    conda env create -f environment.yml --name "$ENV_NAME"
fi

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

echo "Environment '$ENV_NAME' is activated. To deactivate, use 'conda deactivate'."
