#!/usr/bin/env python3
"""
Test script for EngramDB MCP Server

This script creates a simple MCP client that connects to the EngramDB MCP server
and tests its functionality.
"""

import asyncio
import json
import logging
import os
import sys
import uuid
from typing import Dict, Any, List

from mcp.client import Client
from mcp.client.http import http_client

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("engramdb-mcp-test")

async def test_server():
    """Test the EngramDB MCP server's functionality."""
    # Get server URL from environment variable or use default
    server_url = os.environ.get("MCP_SERVER_URL", "http://localhost:8080")
    
    logger.info(f"Connecting to EngramDB MCP server at {server_url}...")
    
    # Connect to the server
    async with http_client(server_url) as client:
        # Initialize the client
        await client.initialize()
        
        # Get available tools
        tools = await client.list_tools()
        logger.info(f"Available tools: {[tool.name for tool in tools.tools]}")
        
        # Test creating a memory
        logger.info("Testing create_memory tool...")
        create_result = await client.call_tool(
            "create_memory",
            {
                "content": "This is a test memory created via MCP",
                "metadata": {
                    "source": "mcp_test",
                    "importance": "high"
                }
            }
        )
        logger.info(f"Create memory result: {create_result}")
        
        # Get the memory ID from the result
        memory_id = create_result.get("id")
        if not memory_id:
            logger.error("Failed to create memory")
            return
        
        # Test retrieving the memory
        logger.info(f"Testing retrieve_memory tool with ID {memory_id}...")
        retrieve_result = await client.call_tool(
            "retrieve_memory",
            {
                "id": memory_id
            }
        )
        logger.info(f"Retrieve memory result: {retrieve_result}")
        
        # Test listing memories
        logger.info("Testing list_memories tool...")
        list_result = await client.call_tool(
            "list_memories",
            {
                "limit": 5
            }
        )
        logger.info(f"List memories result: {list_result}")
        
        # Test searching memories
        logger.info("Testing search_memories tool...")
        search_result = await client.call_tool(
            "search_memories",
            {
                "query": "test memory",
                "limit": 3
            }
        )
        logger.info(f"Search memories result: {search_result}")
        
        # Create another memory for relationship testing
        logger.info("Creating another memory for relationship testing...")
        create_result2 = await client.call_tool(
            "create_memory",
            {
                "content": "This is a related test memory",
                "metadata": {
                    "source": "mcp_test",
                    "importance": "medium"
                }
            }
        )
        memory_id2 = create_result2.get("id")
        logger.info(f"Created second memory with ID {memory_id2}")
        
        # Test creating a relationship
        logger.info("Testing create_relationship tool...")
        relationship_result = await client.call_tool(
            "create_relationship",
            {
                "source_id": memory_id,
                "target_id": memory_id2,
                "relationship_type": "SIMILAR_TO",
                "metadata": {
                    "confidence": 0.85
                }
            }
        )
        logger.info(f"Create relationship result: {relationship_result}")
        
        # Test getting related memories
        logger.info("Testing get_related_memories tool...")
        related_result = await client.call_tool(
            "get_related_memories",
            {
                "id": memory_id,
                "direction": "both"
            }
        )
        logger.info(f"Get related memories result: {related_result}")
        
        # Test deleting memories
        logger.info("Testing delete_memory tool...")
        delete_result = await client.call_tool(
            "delete_memory",
            {
                "id": memory_id
            }
        )
        logger.info(f"Delete memory result: {delete_result}")
        
        delete_result2 = await client.call_tool(
            "delete_memory",
            {
                "id": memory_id2
            }
        )
        logger.info(f"Delete second memory result: {delete_result2}")
        
        logger.info("All tests completed successfully!")

if __name__ == "__main__":
    # Make sure server is running in HTTP mode first
    print("Make sure the EngramDB MCP server is running in HTTP mode before running this test!")
    print("Run: uv --directory . run server.py --http")
    print("Press Ctrl+C to cancel or Enter to continue...")
    try:
        input()
    except KeyboardInterrupt:
        sys.exit(0)
    
    asyncio.run(test_server())