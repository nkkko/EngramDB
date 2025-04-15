#!/bin/bash

# Ensure the script is executable
# chmod +x run_mcp_server.sh

# Redirect all pip output to stderr
echo "Installing dependencies..." >&2
uv pip install modelcontextprotocol >&2 || python3 -m pip install modelcontextprotocol >&2

# Try to install EngramDB from local wheel file
if [ -f "engramdb_py-0.1.0-cp313-cp313-macosx_11_0_arm64.whl" ]; then
    echo "Installing EngramDB from local wheel file..." >&2
    uv pip install ./engramdb_py-0.1.0-cp313-cp313-macosx_11_0_arm64.whl >&2 || python3 -m pip install ./engramdb_py-0.1.0-cp313-cp313-macosx_11_0_arm64.whl >&2
else
    echo "Local wheel file not found, using mock implementation" >&2
fi

# Set environment variables
export PYTHONUNBUFFERED=1
export ENGRAMDB_PATH="${ENGRAMDB_PATH:-engramdb_data}"

# Echo startup information to stderr
echo "Starting EngramDB MCP STDIO Server..." >&2
echo "Using EngramDB data path: $ENGRAMDB_PATH" >&2

# Run the server
exec uv --directory . run server.py