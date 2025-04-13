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
import mcp.types as types

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("engramdb-mcp")

# Create EngramDB database instance
DB_PATH = os.environ.get("ENGRAMDB_PATH", "engramdb_data")
db = engramdb_py.ThreadSafeDatabase(DB_PATH)

# Create MCP server
app = Server(
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
    """Run the MCP server."""
    logger.info("Starting EngramDB MCP Server...")
    
    import mcp.server.stdio
    async with mcp.server.stdio.stdio_server() as streams:
        await app.run(
            streams[0],
            streams[1],
            app.create_initialization_options(
                capabilities={
                    "tools": {},
                }
            )
        )

if __name__ == "__main__":
    asyncio.run(main())