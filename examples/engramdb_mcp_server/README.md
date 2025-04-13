# EngramDB MCP Server

This directory contains a Model Context Protocol (MCP) server implementation for EngramDB, allowing LLM applications to interact with an EngramDB database.

## What is MCP?

The Model Context Protocol (MCP) is an open protocol for enabling LLMs to interact with external tools and data. It allows AI applications like Claude Desktop, Claude Code, Cursor, and others to seamlessly integrate with external services like EngramDB.

## Features

This MCP server provides the following tools for interacting with EngramDB:

- **create_memory**: Create a new memory node in EngramDB
- **retrieve_memory**: Retrieve a memory by ID 
- **search_memories**: Search memories by content (uses vector search)
- **list_memories**: List memories with pagination
- **delete_memory**: Delete a memory
- **create_relationship**: Create a relationship between two memories
- **get_related_memories**: Get memories related to a specific memory

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) - Fast Python package installer and resolver
- EngramDB Python library

## Installation

The server is designed to work with uv for dependency management. If you don't have uv installed:

```bash
# Install uv
curl -fsS https://astral.sh/uv/install.sh | bash
```

### Using uv (Recommended)

uv will automatically handle dependencies and virtual environments:

```bash
# Run the server directly (will set up environment automatically)
uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py
```

### Manual Installation (Alternative)

If you prefer to set up dependencies manually:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install modelcontextprotocol

# Install EngramDB from local source
pip install -e ../../python/
```

## Usage

### Starting the MCP Server (Stdio)

For direct use with applications like Claude Desktop:

```bash
# Using uv (recommended)
uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py

# Or without uv
python server.py
```

### Starting the HTTP Server

For web-based clients or applications that support HTTP/SSE connections:

```bash
# Using uv (recommended)
uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py --http [PORT]

# Or without uv
python server.py --http [PORT]
```

By default, the HTTP server runs on port 8080. You can change this by providing a port number or setting the `MCP_PORT` environment variable:

```bash
MCP_PORT=9000 uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py --http
```

### Configuring the Database Location

By default, EngramDB data is stored in the `engramdb_data` directory. You can change this by setting the `ENGRAMDB_PATH` environment variable:

```bash
ENGRAMDB_PATH=/path/to/data uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py
```

## Example Queries

Here are some examples of how to use the tools provided by this MCP server:

### Creating a Memory

```
Create a new memory with the content "This is important information that I want to remember" and add metadata with a tag "important".
```

### Searching Memories

```
Search for memories related to "important information".
```

### Creating Relationships

```
Create a relationship between memory X and memory Y indicating that X is similar to Y.
```

## Integrating with MCP Clients

### Claude Desktop

#### Setting Up in Claude Desktop

1. In Claude Desktop:
   - Go to Settings > Servers
   - Click "Add Server"
   - Select "Manual Configuration"
   - Copy and paste the contents of `claude_mcp_config.json`
   - Or set up from scratch using the instructions below

#### Manual JSON Configuration

Claude Desktop supports adding MCP servers via JSON. A sample `claude_mcp_config.json` is included in this repository:

```json
{
    "mcpServers": {
        "engramdb-server": {
            "command": "/path/to/uv",
            "args": [
                "--directory",
                "/path/to/engramdb/examples/engramdb_mcp_server",
                "run",
                "server.py"
            ],
            "env": {
                "PYTHONUNBUFFERED": "1",
                "ENGRAMDB_PATH": "/path/to/engramdb_data",
                "PATH": "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
            },
            "logFile": "~/Library/Logs/Claude/mcp-server-engramdb.log"
        }
    }
}
```

Configuration locations:
- macOS: `~/Library/Application Support/Claude/config.json`
- Windows: `%APPDATA%\Claude\config.json`
- Linux: `~/.config/Claude/config.json`

#### Individual Server Configuration

If you prefer to add just this server to your existing configuration:

1. Open Claude Desktop settings > Advanced
2. Click "Edit Configuration" 
3. Add the EngramDB server to the existing `mcpServers` object:

```json
"engramdb-server": {
    "command": "/path/to/uv",
    "args": [
        "--directory",
        "/path/to/engramdb/examples/engramdb_mcp_server",
        "run",
        "server.py"
    ],
    "env": {
        "PYTHONUNBUFFERED": "1",
        "ENGRAMDB_PATH": "/path/to/engramdb_data"
    },
    "logFile": "~/Library/Logs/Claude/mcp-server-engramdb.log"
}
```

### Claude Code and Other CLI Tools

To use the EngramDB MCP server with Claude Code and other command-line based MCP clients:

1. Start the HTTP server: 
   ```bash
   uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py --http
   ```

2. Configure your MCP client to connect to: `http://localhost:8080`

For Claude Code, you can start it with:

```bash
claude-code --server http://localhost:8080
```

### Web-Based & Other MCP Clients

For web-based or other MCP clients that support HTTP/SSE connections:

1. Start the HTTP server: 
   ```bash
   uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py --http
   ```
   
2. Configure your client to connect to: `http://localhost:8080`

Many MCP clients require a JSON configuration. Here's a generic configuration template you can adapt:

```json
{
  "name": "EngramDB Memory Server",
  "id": "engramdb-server",
  "url": "http://localhost:8080"
}
```

## Troubleshooting

If you encounter issues connecting:

1. **Check uv installation**:
   ```bash
   uv --version
   ```

2. **Make sure paths are correct**:
   - Update the path to `uv` in your configuration: run `which uv` to find it
   - Update the directory path to point to the correct location

3. **Check logs**:
   - Look at the log file: `~/Library/Logs/Claude/mcp-server-engramdb.log`
   - Watch for "invalid JSON" errors that indicate stdout/stderr mixing issues

4. **Test connectivity**:
   Run the HTTP server manually and test:
   ```bash
   uv --directory /path/to/engramdb/examples/engramdb_mcp_server run server.py --http
   curl http://localhost:8080
   ```

5. **Check environment**:
   - Make sure `PYTHONUNBUFFERED=1` is set to prevent buffering issues
   - Ensure EngramDB is properly installed and accessible

## Testing

You can test the functionality of the MCP server using the included test script:

```bash
uv --directory /path/to/engramdb/examples/engramdb_mcp_server run test_mcp_server.py
```

This script will connect to the server and test all available tools to verify they're working correctly.

## License

This project is licensed under the same license as EngramDB.