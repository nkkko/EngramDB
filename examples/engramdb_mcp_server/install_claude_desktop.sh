#!/bin/bash
# This script installs the EngramDB MCP Server in Claude Desktop using the MCP CLI

# Get the current directory as the server directory
SERVER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Check if the virtual environment exists
if [ ! -d ".venv" ]; then
  echo "Virtual environment .venv not found. Please run the setup script first."
  exit 1
fi

# Activate the virtual environment
source .venv/bin/activate

# Install the server in Claude Desktop
echo "Installing EngramDB MCP Server in Claude Desktop..."
mcp install "${SERVER_DIR}/mcp_server.py" --name "EngramDB Memory Server" -v ENGRAMDB_PATH="${SERVER_DIR}/engramdb_data" -v PYTHONUNBUFFERED=1

echo ""
echo "EngramDB MCP Server has been installed in Claude Desktop."
echo "You can now use it in the Claude Desktop application."