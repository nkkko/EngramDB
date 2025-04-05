#!/bin/bash
# Script to install EngramDB Python library

# Function to handle script termination
cleanup() {
    echo "Cleaning up installation process..."
    # Return to original directory if interrupted
    cd "$ORIGINAL_DIR"
    exit 0
}

# Set up trap to call cleanup function when script is terminated
trap cleanup SIGINT SIGTERM

# Store original directory
ORIGINAL_DIR=$(pwd)

# Activate virtual environment
source venv_app/bin/activate

# Navigate to the parent directory where the Python package is located
cd ..

# Install the EngramDB Python package in development mode
echo "Installing EngramDB Python package..."
cd python
pip install -e .

# Return to the web directory
cd "$ORIGINAL_DIR"

echo "EngramDB Python package installed!"
echo "You can now run the web interface with the actual implementation using ./run_web.sh"

# Clean up at end of script
cleanup