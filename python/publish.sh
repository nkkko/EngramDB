#!/bin/bash
# Script to build and publish EngramDB Python package

set -e

# Ensure we're in the right directory
cd "$(dirname "$0")"

# Create and activate virtual environment if needed
if [ ! -d "venv" ]; then
  python -m venv venv
fi
source venv/bin/activate

# Install dependencies
pip install -U pip maturin

# Build wheel
echo "Building package..."
maturin build --release

# Publish to PyPI using .pypirc credentials
echo "Publishing to PyPI..."
maturin publish --no-sdist

echo "Package published successfully!"