"""
EngramDB database client for interacting with a specific database.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from .exceptions import ApiError
from .memory_node import MemoryNode
from .models import AttributeFilter, SearchQuery, SearchResult

logger = logging.getLogger(__name__)


class EngramDatabase:
    """
    Client for interacting with a specific EngramDB database.
    
    This class provides methods for interacting with memory nodes and performing searches.
    """
    
    def __init__(
        self,
        client: "EngramClient",  # type: ignore # Forward reference
        database_id: str,
        name: str,
    ):
        """
        Initialize the database client.
        
        Args:
            client: The EngramDB client to use for API requests.
            database_id: The ID of the database.
            name: The name of the database.
        """
        self.client = client
        self.database_id = database_id
        self.name = name
    
    def create_node(
        self,
        vector: Union[List[float], np.ndarray],
        attributes: Optional[Dict[str, Any]] = None,
        connections: Optional[List[Dict[str, Any]]] = None,
        content: Optional[str] = None,
    ) -> MemoryNode:
        """
        Create a new memory node in the database.
        
        Args:
            vector: The embedding vector for the memory node.
            attributes: Optional attributes for the memory node.
            connections: Optional connections to other memory nodes.
            content: Optional text content associated with the memory node.
            
        Returns:
            The created memory node.
        """
        # Convert numpy array to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
            
        data = {
            "vector": vector,
            "attributes": attributes or {},
        }
        
        if connections:
            data["connections"] = connections
            
        if content:
            data["content"] = content
            
        response = self.client._request(
            "POST",
            f"databases/{self.database_id}/nodes",
            json_data=data,
        )
        
        return MemoryNode(
            database=self,
            node_id=response["id"],
            vector=vector,
            attributes=response.get("attributes", {}),
            connections=response.get("connections", []),
            content=response.get("content"),
            created_at=response.get("created_at"),
            updated_at=response.get("updated_at"),
        )
    
    def create_node_from_content(
        self,
        content: str,
        model: str = "default",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> MemoryNode:
        """
        Create a new memory node from text content.
        
        This will use the embeddings API to generate an embedding for the content.
        
        Args:
            content: The text content to embed.
            model: The embedding model to use.
            attributes: Optional attributes for the memory node.
            
        Returns:
            The created memory node.
        """
        data = {
            "content": content,
            "model": model,
        }
        
        if attributes:
            data["attributes"] = attributes
            
        response = self.client._request(
            "POST",
            f"databases/{self.database_id}/nodes/from_content",
            json_data=data,
        )
        
        return MemoryNode(
            database=self,
            node_id=response["id"],
            vector=response.get("vector"),
            attributes=response.get("attributes", {}),
            connections=response.get("connections", []),
            content=response.get("content"),
            created_at=response.get("created_at"),
            updated_at=response.get("updated_at"),
        )
    
    def get_node(
        self,
        node_id: str,
        include_vectors: bool = False,
        include_connections: bool = False,
    ) -> MemoryNode:
        """
        Get a memory node by ID.
        
        Args:
            node_id: The ID of the memory node.
            include_vectors: Whether to include vector embeddings in the response.
            include_connections: Whether to include connections in the response.
            
        Returns:
            The memory node.
            
        Raises:
            ApiError: If the memory node does not exist.
        """
        params = {
            "include_vectors": str(include_vectors).lower(),
            "include_connections": str(include_connections).lower(),
        }
        
        response = self.client._request(
            "GET",
            f"databases/{self.database_id}/nodes/{node_id}",
            params=params,
        )
        
        return MemoryNode(
            database=self,
            node_id=response["id"],
            vector=response.get("vector"),
            attributes=response.get("attributes", {}),
            connections=response.get("connections", []),
            content=response.get("content"),
            created_at=response.get("created_at"),
            updated_at=response.get("updated_at"),
        )
    
    def delete_node(self, node_id: str) -> bool:
        """
        Delete a memory node.
        
        Args:
            node_id: The ID of the memory node to delete.
            
        Returns:
            True if the node was deleted, False otherwise.
        """
        self.client._request(
            "DELETE",
            f"databases/{self.database_id}/nodes/{node_id}",
        )
        return True
    
    def list_nodes(
        self,
        limit: int = 100,
        offset: int = 0,
        include_vectors: bool = False,
        include_connections: bool = False,
    ) -> List[MemoryNode]:
        """
        List memory nodes in the database.
        
        Args:
            limit: Maximum number of nodes to return.
            offset: Number of nodes to skip.
            include_vectors: Whether to include vector embeddings in the response.
            include_connections: Whether to include connections in the response.
            
        Returns:
            A list of memory nodes.
        """
        params = {
            "limit": limit,
            "offset": offset,
            "include_vectors": str(include_vectors).lower(),
            "include_connections": str(include_connections).lower(),
        }
        
        response = self.client._request(
            "GET",
            f"databases/{self.database_id}/nodes",
            params=params,
        )
        
        nodes = response.get("nodes", [])
        
        return [
            MemoryNode(
                database=self,
                node_id=node["id"],
                vector=node.get("vector"),
                attributes=node.get("attributes", {}),
                connections=node.get("connections", []),
                content=node.get("content"),
                created_at=node.get("created_at"),
                updated_at=node.get("updated_at"),
            )
            for node in nodes
        ]
    
    def search(
        self,
        vector: Optional[Union[List[float], np.ndarray]] = None,
        content: Optional[str] = None,
        model: str = "default",
        filters: Optional[List[AttributeFilter]] = None,
        limit: int = 10,
        threshold: float = 0.0,
        include_vectors: bool = False,
        include_connections: bool = False,
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Search for memory nodes.
        
        Args:
            vector: Optional embedding vector to search for.
            content: Optional text content to search for.
            model: Embedding model to use for content search.
            filters: Optional attribute filters to apply.
            limit: Maximum number of results to return.
            threshold: Minimum similarity threshold for results.
            include_vectors: Whether to include vector embeddings in the results.
            include_connections: Whether to include connections in the results.
            
        Returns:
            A list of tuples containing memory nodes and their similarity scores.
            
        Raises:
            ValueError: If neither vector nor content is provided.
        """
        if vector is None and content is None:
            raise ValueError("Either vector or content must be provided for search")
            
        # Convert numpy array to list if needed
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()
            
        # Prepare the search query
        query = SearchQuery(
            vector=vector,
            content=content,
            model=model,
            filters=filters or [],
            limit=limit,
            threshold=threshold,
            include_vectors=include_vectors,
            include_connections=include_connections,
        )
            
        response = self.client._request(
            "POST",
            f"databases/{self.database_id}/search",
            json_data=query.model_dump(),
        )
        
        results = response.get("results", [])
        
        return [
            (
                MemoryNode(
                    database=self,
                    node_id=result["node"]["id"],
                    vector=result["node"].get("vector"),
                    attributes=result["node"].get("attributes", {}),
                    connections=result["node"].get("connections", []),
                    content=result["node"].get("content"),
                    created_at=result["node"].get("created_at"),
                    updated_at=result["node"].get("updated_at"),
                ),
                result["similarity"],
            )
            for result in results
        ]
    
    def search_text(
        self,
        text: str,
        model: str = "default",
        filters: Optional[List[AttributeFilter]] = None,
        limit: int = 10,
        threshold: float = 0.0,
        include_vectors: bool = False,
        include_connections: bool = False,
    ) -> List[Tuple[MemoryNode, float]]:
        """
        Search for memory nodes using text.
        
        This is a convenience method that searches using the text content.
        
        Args:
            text: The text to search for.
            model: The embedding model to use.
            filters: Optional attribute filters to apply.
            limit: Maximum number of results to return.
            threshold: Minimum similarity threshold for results.
            include_vectors: Whether to include vector embeddings in the results.
            include_connections: Whether to include connections in the results.
            
        Returns:
            A list of tuples containing memory nodes and their similarity scores.
        """
        return self.search(
            vector=None,
            content=text,
            model=model,
            filters=filters,
            limit=limit,
            threshold=threshold,
            include_vectors=include_vectors,
            include_connections=include_connections,
        )
        
    def clear_all(self) -> bool:
        """
        Delete all nodes in the database.
        
        Returns:
            True if successful.
        """
        nodes = self.list_nodes()
        for node in nodes:
            self.delete_node(node.id)
        return True