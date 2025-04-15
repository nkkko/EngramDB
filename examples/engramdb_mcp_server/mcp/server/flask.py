"""
Mock MCP Flask Server

This module provides a simple Flask server for MCP that serves 
tools via HTTP and server-sent events (SSE).
"""

import json
import logging
import sys
import traceback
from typing import Any, Callable, Dict

from flask import Flask, jsonify, request, Response, render_template_string

logger = logging.getLogger("mcp.server.flask")

def flask_server(app, host: str = "0.0.0.0", port: int = 8080):
    """Create and run a Flask server for the MCP app."""
    flask_app = Flask("mcp-server")
    
    # Disable default Flask logging
    flask_app.logger.disabled = True
    log = logging.getLogger('werkzeug')
    log.disabled = True
    
    # Define routes
    
    @flask_app.route('/', methods=['GET'])
    def home():
        """Home page with simple UI for testing."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>EngramDB MCP Server</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                h1 { color: #333; }
                h2 { color: #555; }
                pre { background: #f4f4f4; padding: 10px; border-radius: 5px; }
                .tool { margin-bottom: 20px; border: 1px solid #ddd; padding: 15px; border-radius: 5px; }
                button { padding: 8px 15px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }
                input, textarea { width: 100%; padding: 8px; margin: 5px 0 15px 0; display: inline-block; border: 1px solid #ccc; border-radius: 4px; box-sizing: border-box; }
            </style>
        </head>
        <body>
            <h1>EngramDB MCP Server</h1>
            <p>Server Status: <strong>Running</strong></p>
            <p>Version: {{ version }}</p>
            
            <h2>Available Tools</h2>
            {% for name, tool in tools.items() %}
            <div class="tool">
                <h3>{{ name }}</h3>
                <p>{{ tool.description }}</p>
                <pre>{{ tool.input_schema|tojson(indent=2) }}</pre>
            </div>
            {% endfor %}
            
            <h2>Test Tools</h2>
            <div class="tool">
                <h3>Create Memory</h3>
                <form id="create-memory-form">
                    <label for="content">Content:</label>
                    <textarea id="content" rows="4" required></textarea>
                    <button type="submit">Create Memory</button>
                </form>
                <pre id="create-memory-result"></pre>
            </div>
            
            <script>
                document.getElementById('create-memory-form').addEventListener('submit', async (e) => {
                    e.preventDefault();
                    const content = document.getElementById('content').value;
                    
                    try {
                        const response = await fetch('/tools/create_memory', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ content })
                        });
                        
                        const result = await response.json();
                        document.getElementById('create-memory-result').textContent = JSON.stringify(result, null, 2);
                    } catch (error) {
                        document.getElementById('create-memory-result').textContent = 'Error: ' + error.message;
                    }
                });
            </script>
        </body>
        </html>
        """
        tools = {}
        for name, tool_def in app.tool_definitions.items():
            tools[name] = {
                "name": name,
                "description": tool_def.description,
                "input_schema": tool_def.input_schema
            }
        
        return render_template_string(html, version=app.version, tools=tools)
    
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
    
    # Add SSE endpoint for MCP Inspector
    @flask_app.route('/sse', methods=['GET'])
    def sse():
        """Server-sent events endpoint for MCP Inspector."""
        def generate():
            # Send an initial message
            yield f"data: {json.dumps({'type': 'connection', 'status': 'connected'})}\n\n"
            
            # Keep the connection open
            while True:
                # In a real implementation, you would yield new events as they happen
                import time
                time.sleep(60)  # Keep connection alive
        
        return Response(generate(), mimetype='text/event-stream')
    
    @flask_app.route('/tools/<tool_name>', methods=['POST'])
    async def call_tool(tool_name):
        """Call a specific tool."""
        if tool_name not in app.tools:
            return jsonify({"error": f"Tool '{tool_name}' not found"}), 404
        
        # Get request data
        try:
            data = request.json
            if data is None:
                return jsonify({"error": "Invalid JSON in request"}), 400
            
            logger.info(f"Calling tool {tool_name} with data: {data}")
            
            # Call the tool function
            tool_func = app.tools[tool_name]
            result = await tool_func(**data)
            return jsonify({"result": result})
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            stack_trace = traceback.format_exc()
            logger.error(f"{error_msg}\n{stack_trace}")
            return jsonify({"error": error_msg, "traceback": stack_trace}), 500
    
    # Start the Flask server
    logger.info(f"Starting Flask server on {host}:{port}")
    flask_app.run(host=host, port=port, debug=False, threaded=True)
    
    return flask_app