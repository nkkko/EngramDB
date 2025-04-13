"""
Mock MCP Server module
"""

import asyncio
import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional

from ..types import ToolDefinition

logger = logging.getLogger("mcp.server")

class Server:
    """Mock MCP Server implementation."""
    
    def __init__(self, name: str, version: str, description: str):
        self.name = name
        self.version = version
        self.description = description
        self.tools = {}
        self.tool_definitions = {}
    
    def register_tool(self, name: str, description: str, input_schema: Dict[str, Any] = None):
        """Register a tool with the server."""
        def decorator(func):
            self.tools[name] = func
            self.tool_definitions[name] = ToolDefinition(
                name=name,
                description=description,
                input_schema=input_schema or {}
            )
            return func
        return decorator
    
    async def run(self, input_stream, output_stream, initialization_options: Dict[str, Any] = None):
        """Run the server with the given streams."""
        logger.info(f"Starting MCP server {self.name} v{self.version}")
        try:
            while True:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Server shutting down")
    
    def create_initialization_options(self, capabilities: Dict[str, Any] = None) -> Dict[str, Any]:
        """Create initialization options for the server."""
        return {
            "capabilities": capabilities or {}
        }