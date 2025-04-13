#!/usr/bin/env python3
"""
EngramDB MCP Server (HTTP/SSE)

This script starts an HTTP server that serves the EngramDB MCP API using
Server-Sent Events (SSE) for client-server communication.
"""

import asyncio
import logging
import os
import json
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Try different import approaches for EngramDB
try:
    import engramdb_py
    print("Successfully imported engramdb_py")
except ImportError:
    try:
        import engramdb
        print("Successfully imported engramdb")
        # Use engramdb as engramdb_py
        engramdb_py = engramdb
    except ImportError as e:
        sys.exit(f"Error: Cannot import EngramDB module. Please ensure it is installed correctly. Error: {e}")

from mcp.server import Server
from mcp.server.flask import flask_server
import mcp.types as types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("engramdb-mcp-http")

# Create EngramDB database instance
DB_PATH = os.environ.get("ENGRAMDB_PATH", "engramdb_data")
db = engramdb_py.ThreadSafeDatabase(DB_PATH)

# Import the MCP server app from the main module
from engramdb_mcp_server import app

if __name__ == "__main__":
    # Get port from environment variable or use default
    port = int(os.environ.get("MCP_PORT", 8080))
    
    # Print startup message
    logger.info(f"Starting EngramDB MCP HTTP Server on port {port}...")
    logger.info(f"Using EngramDB database at: {DB_PATH}")
    
    # Start Flask server
    flask_server(app, port=port)