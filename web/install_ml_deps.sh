#!/bin/bash
# Script to install ML dependencies for EngramDB web

# Function to handle script termination
cleanup() {
    echo "Cleaning up installation process..."
    exit 0
}

# Set up trap to call cleanup function when script is terminated
trap cleanup SIGINT SIGTERM

# Create virtual environment if it doesn't exist
if [ ! -d "venv_app" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_app
fi

# Activate virtual environment
source venv_app/bin/activate

# Install base requirements
echo "Installing base requirements..."
pip install -r requirements.txt

# Install ML requirements specifically
echo "Installing ML dependencies..."
pip install torch transformers sentence-transformers

echo "ML dependencies installed successfully!"
echo "You can now run the application with ./run_web.sh"

# Clean up at end of script
cleanup