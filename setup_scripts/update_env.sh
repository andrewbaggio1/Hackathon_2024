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
    echo "Environment name could not be retrieved from config.py."
    exit 1
fi

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate "$ENV_NAME"

# Export the current environment to environment.yml
conda env export --from-history > environment.yml

echo "environment.yml updated with the current environment: '$ENV_NAME'."
