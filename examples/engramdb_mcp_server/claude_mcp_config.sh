#!/bin/bash
# This script creates a Claude Desktop configuration file for the EngramDB MCP server

# Get the current directory as the server directory
SERVER_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Get the path to the Python executable in the virtual environment
if [ -d "${SERVER_DIR}/.venv" ]; then
  PYTHON_PATH="${SERVER_DIR}/.venv/bin/python"
elif [ -d "${SERVER_DIR}/venv" ]; then
  PYTHON_PATH="${SERVER_DIR}/venv/bin/python"
else
  echo "Virtual environment not found. Please create one first."
  exit 1
fi

# Create a config file
CONFIG_FILE="${SERVER_DIR}/claude_config.json"

# Create the JSON configuration
cat > "$CONFIG_FILE" << EOF
{
  "mcpServers": {
    "engramdb-server": {
      "command": "${PYTHON_PATH}",
      "args": [
        "${SERVER_DIR}/mcp_server.py"
      ],
      "env": {
        "PYTHONUNBUFFERED": "1",
        "ENGRAMDB_PATH": "${SERVER_DIR}/engramdb_data",
        "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
      },
      "logFile": "~/Library/Logs/Claude/mcp-server-engramdb.log"
    }
  }
}
EOF

# Print the configuration file path
echo "Generated configuration at: $CONFIG_FILE"
echo ""
echo "Instructions:"
echo "1. Open Claude Desktop"
echo "2. Go to Settings > Servers"
echo "3. Click 'Add Server'"
echo "4. Select 'Manual Configuration'"
echo "5. Copy and paste the contents of $CONFIG_FILE"
echo ""
echo "Configuration contents:"
echo "======================="
cat "$CONFIG_FILE"
echo "======================="