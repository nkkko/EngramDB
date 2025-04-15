#!/usr/bin/env python3
"""
EngramDB MCP Server

This server implements the Model Context Protocol (MCP) for EngramDB,
providing tools to interact with an EngramDB database.
"""

import asyncio
import logging
import os
import json
import sys
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

# Configure logging first to ensure all output goes to stderr
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr  # Redirect logs to stderr to avoid breaking JSON communication
)
logger = logging.getLogger("engramdb-mcp")

# Import modules after logging is configured
try:
    import engramdb as engramdb_py
    logger.info("Successfully imported engramdb module")
except ImportError:
    try:
        import engramdb_py
        logger.info("Successfully imported engramdb_py module")
    except ImportError:
        logger.info("Using mock engramdb_py module for testing")
        # Create a mock module for testing
        import sys
        from pathlib import Path
        
        # Import our local mock module
        sys.path.insert(0, str(Path(__file__).parent.absolute()))
        import engramdb_py

# Create EngramDB database instance
DB_PATH = os.environ.get("ENGRAMDB_PATH", "engramdb_data")
db = engramdb_py.ThreadSafeDatabase(DB_PATH)

class MCPServer:
    """MCP Server for EngramDB operations using JSON-RPC 2.0 format."""
    
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
            self.tool_definitions[name] = {
                "name": name,
                "description": description,
                "input_schema": input_schema or {}
            }
            return func
        return decorator
    
    async def handle_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an incoming MCP message in JSON-RPC 2.0 format."""
        method = message.get("method", "")
        msg_id = message.get("id")
        
        if method == "initialize":
            return await self.handle_initialize(message)
        elif method == "tools/invoke":
            return await self.handle_tool_call(message)
        else:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
    
    async def handle_initialize(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle an initialization message."""
        logger.info("Received initialize message")
        
        # Extract client capabilities
        params = message.get("params", {})
        client_info = params.get("clientInfo", {})
        client_name = client_info.get("name", "Unknown Client")
        client_version = client_info.get("version", "Unknown Version")
        msg_id = message.get("id")
        
        logger.info(f"Client: {client_name} {client_version}")
        
        # Prepare tool definitions for response
        tools = {}
        for name, tool_def in self.tool_definitions.items():
            tools[name] = {
                "name": tool_def["name"],
                "description": tool_def["description"],
                "input_schema": tool_def["input_schema"]
            }
        
        # Respond with server capabilities in JSON-RPC format
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "serverInfo": {
                    "name": self.name,
                    "version": self.version,
                    "description": self.description
                },
                "capabilities": {
                    "tools": {tool: {"schema": self.tool_definitions[tool]["input_schema"]} for tool in self.tool_definitions}
                }
            }
        }
    
    async def handle_tool_call(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a tool call message."""
        params = message.get("params", {})
        tool_name = params.get("name")
        tool_input = params.get("input", {})
        msg_id = message.get("id")
        
        logger.info(f"Received tool call: {tool_name}")
        
        if tool_name not in self.tools:
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32601,
                    "message": f"Tool not found: {tool_name}"
                }
            }
        
        try:
            # Call the tool function with the provided input
            tool_func = self.tools[tool_name]
            result = await tool_func(**tool_input)
            
            # Return successful result in JSON-RPC format
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": result
            }
        except Exception as e:
            # Log the error details
            logger.error(f"Error calling tool {tool_name}: {str(e)}")
            
            # Return error result in JSON-RPC format
            return {
                "jsonrpc": "2.0",
                "id": msg_id,
                "error": {
                    "code": -32000,
                    "message": str(e)
                }
            }

# Create MCP server
app = MCPServer(
    "engramdb-server",
    version="0.1.0",
    description="MCP server for EngramDB operations"
)

# Define tools for EngramDB
@app.register_tool(
    name="create_memory",
    description="Create a new memory node in EngramDB",
    input_schema={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "Content of the memory to store"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata for the memory",
                "additionalProperties": True
            }
        },
        "required": ["content"]
    }
)
async def create_memory(content: str, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Create a new memory in the EngramDB database."""
    if metadata is None:
        metadata = {}
    
    # Add timestamp if not present
    if "timestamp" not in metadata:
        metadata["timestamp"] = datetime.utcnow().isoformat()
    
    # Create memory node
    memory_id = str(uuid.uuid4())
    memory = engramdb_py.MemoryNode(
        id=memory_id,
        content=content,
        metadata=metadata
    )
    
    # Store in database
    db.store_memory(memory)
    
    return {
        "id": memory_id,
        "content": content,
        "metadata": metadata
    }


@app.register_tool(
    name="retrieve_memory",
    description="Retrieve a memory by ID from EngramDB",
    input_schema={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "ID of the memory to retrieve"
            }
        },
        "required": ["id"]
    }
)
async def retrieve_memory(id: str) -> Dict[str, Any]:
    """Retrieve a memory from the EngramDB database by ID."""
    try:
        memory = db.get_memory(id)
        if memory:
            return {
                "id": memory.id,
                "content": memory.content,
                "metadata": memory.metadata
            }
        else:
            return {"error": f"Memory with ID {id} not found"}
    except Exception as e:
        logger.error(f"Error retrieving memory: {e}")
        return {"error": str(e)}


@app.register_tool(
    name="search_memories",
    description="Search memories in EngramDB by content",
    input_schema={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to search for in memory contents"
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "default": 10
            }
        },
        "required": ["query"]
    }
)
async def search_memories(query: str, limit: int = 10) -> Dict[str, Any]:
    """Search for memories in the EngramDB database by content."""
    try:
        # Search using semantic search with the database
        results = db.search_memories(query, limit=limit)
        
        memories = []
        for memory in results:
            memories.append({
                "id": memory.id,
                "content": memory.content,
                "metadata": memory.metadata,
                "score": getattr(memory, "score", None)
            })
        
        return {
            "query": query,
            "results": memories
        }
    except Exception as e:
        logger.error(f"Error searching memories: {e}")
        return {"error": str(e)}


@app.register_tool(
    name="list_memories",
    description="List memories in EngramDB",
    input_schema={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return",
                "default": 10
            },
            "offset": {
                "type": "integer",
                "description": "Number of memories to skip",
                "default": 0
            }
        }
    }
)
async def list_memories(limit: int = 10, offset: int = 0) -> Dict[str, Any]:
    """List memories from the EngramDB database."""
    try:
        # Get all memories
        all_memories = db.get_all_memories()
        
        # Apply offset and limit
        memories = all_memories[offset:offset + limit]
        
        results = []
        for memory in memories:
            results.append({
                "id": memory.id,
                "content": memory.content,
                "metadata": memory.metadata
            })
        
        return {
            "total": len(all_memories),
            "limit": limit,
            "offset": offset,
            "results": results
        }
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        return {"error": str(e)}


@app.register_tool(
    name="delete_memory",
    description="Delete a memory from EngramDB",
    input_schema={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "ID of the memory to delete"
            }
        },
        "required": ["id"]
    }
)
async def delete_memory(id: str) -> Dict[str, Any]:
    """Delete a memory from the EngramDB database."""
    try:
        success = db.delete_memory(id)
        if success:
            return {"success": True, "id": id}
        else:
            return {"success": False, "error": f"Memory with ID {id} not found"}
    except Exception as e:
        logger.error(f"Error deleting memory: {e}")
        return {"error": str(e)}


@app.register_tool(
    name="create_relationship",
    description="Create a relationship between two memories in EngramDB",
    input_schema={
        "type": "object",
        "properties": {
            "source_id": {
                "type": "string",
                "description": "ID of the source memory"
            },
            "target_id": {
                "type": "string",
                "description": "ID of the target memory"
            },
            "relationship_type": {
                "type": "string",
                "description": "Type of relationship between memories",
                "enum": ["REFERENCES", "SIMILAR_TO", "BEFORE", "AFTER", "PARENT_OF", "CHILD_OF", "CUSTOM"]
            },
            "custom_type": {
                "type": "string",
                "description": "Custom relationship type name (required if relationship_type is CUSTOM)"
            },
            "metadata": {
                "type": "object",
                "description": "Optional metadata for the relationship",
                "additionalProperties": True
            }
        },
        "required": ["source_id", "target_id", "relationship_type"]
    }
)
async def create_relationship(
    source_id: str, 
    target_id: str,
    relationship_type: str,
    custom_type: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create a relationship between two memories in the EngramDB database."""
    try:
        if metadata is None:
            metadata = {}
            
        # Add timestamp if not present
        if "timestamp" not in metadata:
            metadata["timestamp"] = datetime.utcnow().isoformat()
        
        # Map string to RelationshipType enum
        rel_type_map = {
            "REFERENCES": engramdb_py.RelationshipType.REFERENCES,
            "SIMILAR_TO": engramdb_py.RelationshipType.SIMILAR_TO,
            "BEFORE": engramdb_py.RelationshipType.BEFORE,
            "AFTER": engramdb_py.RelationshipType.AFTER,
            "PARENT_OF": engramdb_py.RelationshipType.PARENT_OF,
            "CHILD_OF": engramdb_py.RelationshipType.CHILD_OF,
            "CUSTOM": engramdb_py.RelationshipType.CUSTOM
        }
        
        rel_type = rel_type_map.get(relationship_type)
        if rel_type is None:
            return {"error": f"Invalid relationship type: {relationship_type}"}
        
        # For custom types, we need the custom type name
        if rel_type == engramdb_py.RelationshipType.CUSTOM and custom_type is None:
            return {"error": "Custom relationship type requires 'custom_type' parameter"}
        
        # Create relationship
        success = db.create_relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=rel_type,
            custom_type=custom_type,
            metadata=metadata
        )
        
        if success:
            return {
                "success": True,
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
                "custom_type": custom_type,
                "metadata": metadata
            }
        else:
            return {"success": False, "error": "Failed to create relationship"}
    except Exception as e:
        logger.error(f"Error creating relationship: {e}")
        return {"error": str(e)}


@app.register_tool(
    name="get_related_memories",
    description="Get memories related to a specific memory in EngramDB",
    input_schema={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "ID of the memory to find relationships for"
            },
            "relationship_type": {
                "type": "string",
                "description": "Optional filter by relationship type",
                "enum": ["REFERENCES", "SIMILAR_TO", "BEFORE", "AFTER", "PARENT_OF", "CHILD_OF", "CUSTOM"]
            },
            "custom_type": {
                "type": "string",
                "description": "Custom relationship type name (used if relationship_type is CUSTOM)"
            },
            "direction": {
                "type": "string",
                "description": "Direction of relationship (outgoing = from this memory, incoming = to this memory)",
                "enum": ["outgoing", "incoming", "both"],
                "default": "both"
            }
        },
        "required": ["id"]
    }
)
async def get_related_memories(
    id: str,
    relationship_type: Optional[str] = None,
    custom_type: Optional[str] = None,
    direction: str = "both"
) -> Dict[str, Any]:
    """Get memories related to a specific memory in the EngramDB database."""
    try:
        # Map string to RelationshipType enum if provided
        rel_type = None
        if relationship_type:
            rel_type_map = {
                "REFERENCES": engramdb_py.RelationshipType.REFERENCES,
                "SIMILAR_TO": engramdb_py.RelationshipType.SIMILAR_TO,
                "BEFORE": engramdb_py.RelationshipType.BEFORE,
                "AFTER": engramdb_py.RelationshipType.AFTER,
                "PARENT_OF": engramdb_py.RelationshipType.PARENT_OF,
                "CHILD_OF": engramdb_py.RelationshipType.CHILD_OF,
                "CUSTOM": engramdb_py.RelationshipType.CUSTOM
            }
            rel_type = rel_type_map.get(relationship_type)
            if rel_type is None:
                return {"error": f"Invalid relationship type: {relationship_type}"}
        
        # Get related memories based on direction
        if direction == "outgoing" or direction == "both":
            outgoing = db.get_outgoing_relationships(id, relationship_type=rel_type, custom_type=custom_type)
        else:
            outgoing = []
            
        if direction == "incoming" or direction == "both":
            incoming = db.get_incoming_relationships(id, relationship_type=rel_type, custom_type=custom_type)
        else:
            incoming = []
        
        # Format results
        outgoing_results = []
        for rel in outgoing:
            rel_memory = db.get_memory(rel.target_id)
            if rel_memory:
                outgoing_results.append({
                    "id": rel_memory.id,
                    "content": rel_memory.content,
                    "metadata": rel_memory.metadata,
                    "relationship": {
                        "type": rel.relationship_type.name,
                        "custom_type": rel.custom_type,
                        "metadata": rel.metadata
                    }
                })
        
        incoming_results = []
        for rel in incoming:
            rel_memory = db.get_memory(rel.source_id)
            if rel_memory:
                incoming_results.append({
                    "id": rel_memory.id,
                    "content": rel_memory.content,
                    "metadata": rel_memory.metadata,
                    "relationship": {
                        "type": rel.relationship_type.name,
                        "custom_type": rel.custom_type,
                        "metadata": rel.metadata
                    }
                })
        
        return {
            "memory_id": id,
            "outgoing_relationships": outgoing_results,
            "incoming_relationships": incoming_results
        }
    except Exception as e:
        logger.error(f"Error getting related memories: {e}")
        return {"error": str(e)}


async def main():
    """Main function to run the STDIO MCP server."""
    logger.info("Starting EngramDB MCP Server (STDIO)")
    logger.info(f"Using EngramDB database at: {DB_PATH}")
    
    # Set up stdin/stdout for reading/writing
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    
    writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
        asyncio.streams.FlowControlMixin, sys.stdout
    )
    writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())
    
    try:
        while True:
            # Read a line from stdin
            line = await reader.readline()
            if not line:  # EOF
                break
                
            # Parse the JSON message
            try:
                message = json.loads(line.decode('utf-8'))
                logger.info(f"Received message: {message.get('method')}")
                
                # Handle the message
                response = await app.handle_message(message)
                
                # Write the response
                response_json = json.dumps(response) + "\n"
                writer.write(response_json.encode('utf-8'))
                await writer.drain()
                
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON: {line.decode('utf-8')}")
                error_response = json.dumps({
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": "Parse error"
                    }
                }) + "\n"
                writer.write(error_response.encode('utf-8'))
                await writer.drain()
                
    except KeyboardInterrupt:
        logger.info("Server shutting down")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        writer.close()
        logger.info("Server stopped")


if __name__ == "__main__":
    # Run in STDIO mode only as requested
    asyncio.run(main())