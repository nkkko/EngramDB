#!/bin/bash

# Redirect all pip output to stderr
/Users/nikola/.pyenv/shims/python3 -m pip install modelcontextprotocol 1>&2
/Users/nikola/.pyenv/shims/python3 -m pip install -e /Users/nikola/dev/engramdb/python/ 1>&2

# Create a virtual environment if needed
if [ ! -d "venv" ]; then
    /Users/nikola/.pyenv/shims/python3 -m venv venv 1>&2
    echo "Created virtual environment" 1>&2
fi

# Run the server with the virtual environment python
export PYTHONUNBUFFERED=1
exec ./venv/bin/python /Users/nikola/dev/engramdb/examples/engramdb_mcp_server/engramdb_mcp_server.py