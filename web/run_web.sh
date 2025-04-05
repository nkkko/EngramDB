#!/bin/bash

# Function to handle script termination
cleanup() {
    echo "Stopping web server..."
    # Find and kill any Python processes running the Flask app
    pkill -f "python app_full.py" || true
    exit 0
}

# Set up trap to call cleanup function when script is terminated
trap cleanup SIGINT SIGTERM

# Create and activate a virtual environment if it doesn't exist
if [ ! -d "venv_app" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv_app
fi

# Activate the virtual environment
source venv_app/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install flask==2.3.3 werkzeug==2.3.7 flask-wtf==1.2.1

# Inform user that we're using mock implementation for numpy
echo "Using mock implementation for vector operations."

# Run the application
echo "Starting the EngramDB web application on port 8082..."
echo "Press Ctrl+C to stop the server gracefully."
python app_full.py &
FLASK_PID=$!

# Wait for the Flask process to finish
wait $FLASK_PID

# If we reach here, the Flask process has ended naturally
cleanup