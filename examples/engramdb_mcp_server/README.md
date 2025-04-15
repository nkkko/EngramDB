# EngramDB MCP Server

This directory contains a Model Context Protocol (MCP) server implementation for EngramDB, allowing LLM applications to interact with an EngramDB database over STDIO.

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

### Using uv (Required)

uv handles dependencies and virtual environments efficiently:

```bash
# Install required dependencies
uv pip install -r requirements.txt

# Install EngramDB (either from PyPI or from the provided wheel file)
uv pip install engramdb-py
# or
uv pip install ./engramdb_py-0.1.0-cp313-cp313-macosx_11_0_arm64.whl

# Run the server directly
uv --directory . run server.py
```

## Usage

### Starting the MCP Server (STDIO)

For direct use with applications like Claude Desktop:

```bash
# Run the server with the provided script
./run_mcp_server.sh

# Or run it directly
uv --directory . run server.py
```

### Using with Claude Desktop

#### Setting Up in Claude Desktop

##### Manual Configuration

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

For CLI tools that support STDIO-based MCP servers, you can either:

1. Use the included script:
   ```bash
   ./run_mcp_server.sh
   ```

2. Or start the server directly:
   ```bash
   uv --directory . run server.py
   ```

## Configuring the Database Location

By default, EngramDB data is stored in the `engramdb_data` directory. You can change this by setting the `ENGRAMDB_PATH` environment variable:

```bash
ENGRAMDB_PATH=/path/to/data uv --directory . run server.py
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

## Troubleshooting

If you encounter issues connecting:

1. **Check dependencies**:
   ```bash
   # Check uv installation
   uv --version
   ```

2. **Make sure paths are correct**:
   - Update the path to `uv` in your configuration: run `which uv` to find it
   - Update the directory path to point to the correct location

3. **Check logs**:
   - Look at the log file: `~/Library/Logs/Claude/mcp-server-engramdb.log`
   - Watch for "invalid JSON" errors that indicate stdout/stderr mixing issues

4. **Check environment**:
   - Make sure `PYTHONUNBUFFERED=1` is set to prevent buffering issues
   - Ensure EngramDB is properly installed and accessible

## License

This project is licensed under the same license as EngramDB.