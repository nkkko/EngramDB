# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build and Run Commands
- Run server (stdio mode): `uv --directory . run server.py`
- Run server (HTTP mode): `uv --directory . run server.py --http [PORT]`
- Run with environment variables: `ENGRAMDB_PATH=/path/to/data uv --directory . run server.py`
- Run simplified server: `mcp run simple_server.py`
- Test: `uv --directory . run test_mcp_server.py`

## Project Structure
- `server.py`: Primary MCP server implementation (supports both stdio and HTTP)
- `simple_server.py`: Simplified implementation using FastMCP API
- `engramdb_py.py`: Mock implementation of EngramDB for testing
- `requirements.txt`: Project dependencies
- `mcp/`: Local MCP server implementation modules

## Installation
- Install dependencies: `uv pip install -r requirements.txt`
- Install EngramDB from local source: `uv pip install -e ../../python/`
- Always use uv (Astral) to manage project dependencies

## Code Style Guidelines
- Follow PEP 8 conventions
- Use descriptive variable names in snake_case
- Class names in PascalCase
- Use type hints for all function parameters and return values
- Organize imports: standard library first, then third-party libs, then local modules
- Use docstrings for all functions and classes
- Use f-strings for string formatting
- Handle errors with try/except blocks with detailed error messages
- Use async/await syntax for asynchronous code
- Use JSONSchema for MCP tool input validation