#!/bin/bash
# Script to run the embedding example with proper cleanup

# Function to handle script termination
cleanup() {
    echo "Cleaning up..."
    # Kill any python processes created by this script if needed
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

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Run the example
echo "Running embedding example..."
python examples/embedding_example.py

# Clean up after the script finishes
cleanup