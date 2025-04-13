"""
Mock MCP Flask Server

This module provides a simple Flask server for MCP that serves 
tools via HTTP and server-sent events (SSE).
"""

import json
import logging
import sys
from typing import Any, Callable, Dict

from flask import Flask, jsonify, request, Response

logger = logging.getLogger("mcp.server.flask")

def flask_server(app, host: str = "0.0.0.0", port: int = 8080):
    """Create and run a Flask server for the MCP app."""
    flask_app = Flask("mcp-server")
    
    # Disable default Flask logging
    flask_app.logger.disabled = True
    log = logging.getLogger('werkzeug')
    log.disabled = True
    
    # Define routes
    
    @flask_app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "ok",
            "version": app.version,
            "name": app.name
        })
    
    @flask_app.route('/tools', methods=['GET'])
    def list_tools():
        """List available tools."""
        tools = {}
        for name, tool_def in app.tool_definitions.items():
            tools[name] = {
                "name": name,
                "description": tool_def.description,
                "input_schema": tool_def.input_schema
            }
        return jsonify({"tools": tools})
    
    @flask_app.route('/tools/<tool_name>', methods=['POST'])
    async def call_tool(tool_name):
        """Call a specific tool."""
        if tool_name not in app.tools:
            return jsonify({"error": f"Tool '{tool_name}' not found"}), 404
        
        # Get request data
        data = request.json
        
        try:
            # Call the tool function
            tool_func = app.tools[tool_name]
            result = await tool_func(**data)
            return jsonify({"result": result})
        except Exception as e:
            logger.error(f"Error calling tool {tool_name}: {e}")
            return jsonify({"error": str(e)}), 500
    
    # Start the Flask server
    logger.info(f"Starting Flask server on {host}:{port}")
    flask_app.run(host=host, port=port, debug=False, threaded=True)
    
    return flask_app