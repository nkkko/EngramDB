"""
Memory node class for the EngramDB client.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, cast

import numpy as np

from .exceptions import ApiError

logger = logging.getLogger(__name__)


class MemoryNode:
    """
    A memory node in the EngramDB database.
    
    This class represents a memory node with vector embeddings, attributes, and connections.
    """
    
    def __init__(
        self,
        database: "EngramDatabase",  # type: ignore # Forward reference
        node_id: str,
        vector: Optional[Union[List[float], np.ndarray]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        connections: Optional[List[Dict[str, Any]]] = None,
        content: Optional[str] = None,
        created_at: Optional[Union[str, datetime]] = None,
        updated_at: Optional[Union[str, datetime]] = None,
    ):
        """
        Initialize a memory node.
        
        Args:
            database: The database this node belongs to.
            node_id: The ID of the node.
            vector: Optional embedding vector for the node.
            attributes: Optional attributes for the node.
            connections: Optional connections to other nodes.
            content: Optional text content associated with the node.
            created_at: Optional creation timestamp.
            updated_at: Optional update timestamp.
        """
        self.database = database
        self.id = node_id
        
        # Handle vector (convert to numpy array if provided)
        if vector is not None:
            self._vector = np.array(vector, dtype=np.float32)
        else:
            self._vector = None
            
        self._attributes = attributes or {}
        self._connections = connections or []
        self._content = content
        
        # Handle timestamps
        if isinstance(created_at, str):
            self._created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        else:
            self._created_at = created_at or datetime.now()
            
        if isinstance(updated_at, str):
            self._updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
        else:
            self._updated_at = updated_at or datetime.now()
            
        # Flag to track local changes
        self._modified = False
    
    def vector(self) -> Optional[np.ndarray]:
        """
        Get the vector embedding for this node.
        
        Returns:
            The vector embedding as a numpy array, or None if not available.
        """
        return self._vector
    
    def set_vector(self, vector: Union[List[float], np.ndarray]) -> None:
        """
        Set the vector embedding for this node.
        
        Args:
            vector: The new vector embedding.
        """
        self._vector = np.array(vector, dtype=np.float32)
        self._modified = True
    
    def attributes(self) -> Dict[str, Any]:
        """
        Get all attributes for this node.
        
        Returns:
            A dictionary of attributes.
        """
        return self._attributes
    
    def get_attribute(self, key: str) -> Optional[Any]:
        """
        Get the value of an attribute.
        
        Args:
            key: The attribute key.
            
        Returns:
            The attribute value, or None if the attribute does not exist.
        """
        return self._attributes.get(key)
    
    def set_attribute(self, key: str, value: Any) -> None:
        """
        Set an attribute value.
        
        Args:
            key: The attribute key.
            value: The attribute value.
        """
        self._attributes[key] = value
        self._modified = True
    
    def remove_attribute(self, key: str) -> None:
        """
        Remove an attribute.
        
        Args:
            key: The attribute key to remove.
        """
        if key in self._attributes:
            del self._attributes[key]
            self._modified = True
    
    def connections(self) -> List[Dict[str, Any]]:
        """
        Get all connections for this node.
        
        Returns:
            A list of connections.
        """
        return self._connections
    
    def add_connection(
        self,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0,
        custom_type: Optional[str] = None,
    ) -> None:
        """
        Add a connection to another node.
        
        This method only adds the connection locally. Call save() to persist the changes.
        
        Args:
            target_id: The ID of the target node.
            relationship_type: The type of relationship.
            strength: The strength of the connection (0.0 to 1.0).
            custom_type: Optional custom relationship type name (for custom types).
        """
        connection = {
            "target_id": target_id,
            "type_name": relationship_type,
            "strength": strength,
        }
        
        if custom_type:
            connection["custom_type"] = custom_type
            
        self._connections.append(connection)
        self._modified = True
    
    def remove_connection(self, target_id: str) -> bool:
        """
        Remove a connection to another node.
        
        This method only removes the connection locally. Call save() to persist the changes.
        
        Args:
            target_id: The ID of the target node.
            
        Returns:
            True if the connection was removed, False if it did not exist.
        """
        initial_length = len(self._connections)
        self._connections = [
            conn for conn in self._connections if conn["target_id"] != target_id
        ]
        
        removed = len(self._connections) < initial_length
        if removed:
            self._modified = True
            
        return removed
    
    def content(self) -> Optional[str]:
        """
        Get the text content associated with this node.
        
        Returns:
            The text content, or None if not available.
        """
        return self._content
    
    def set_content(self, content: str) -> None:
        """
        Set the text content associated with this node.
        
        Args:
            content: The new text content.
        """
        self._content = content
        self._modified = True
    
    def created_at(self) -> datetime:
        """
        Get the creation timestamp.
        
        Returns:
            The creation timestamp.
        """
        return self._created_at
    
    def updated_at(self) -> datetime:
        """
        Get the last update timestamp.
        
        Returns:
            The last update timestamp.
        """
        return self._updated_at
    
    def save(self) -> None:
        """
        Save changes to this node to the database.
        
        Raises:
            ApiError: If the API returns an error response.
        """
        if not self._modified:
            return
            
        data = {
            "attributes": self._attributes,
            "connections": self._connections,
        }
        
        if self._vector is not None:
            data["vector"] = self._vector.tolist()
            
        if self._content is not None:
            data["content"] = self._content
            
        response = self.database.client._request(
            "PUT",
            f"databases/{self.database.database_id}/nodes/{self.id}",
            json_data=data,
        )
        
        # Update local state with server response
        if "vector" in response:
            self._vector = np.array(response["vector"], dtype=np.float32)
            
        self._attributes = response.get("attributes", {})
        self._connections = response.get("connections", [])
        self._content = response.get("content")
        
        if "updated_at" in response:
            if isinstance(response["updated_at"], str):
                self._updated_at = datetime.fromisoformat(
                    response["updated_at"].replace("Z", "+00:00")
                )
            else:
                self._updated_at = response["updated_at"]
                
        self._modified = False
    
    def refresh(self) -> None:
        """
        Refresh the node from the database.
        
        Raises:
            ApiError: If the API returns an error response.
        """
        node = self.database.get_node(
            self.id,
            include_vectors=True,
            include_connections=True,
        )
        
        self._vector = node.vector()
        self._attributes = node.attributes()
        self._connections = node.connections()
        self._content = node.content()
        self._created_at = node.created_at()
        self._updated_at = node.updated_at()
        self._modified = False
    
    def delete(self) -> bool:
        """
        Delete this node from the database.
        
        Returns:
            True if the node was deleted successfully.
            
        Raises:
            ApiError: If the API returns an error response.
        """
        return self.database.delete_node(self.id)
    
    def get_connections(
        self,
        relationship_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get connections from this node.
        
        This method gets connections from the API, not just locally cached ones.
        
        Args:
            relationship_type: Optional relationship type to filter by.
            
        Returns:
            A list of connections.
            
        Raises:
            ApiError: If the API returns an error response.
        """
        params = {}
        if relationship_type:
            params["relationship_type"] = relationship_type
            
        response = self.database.client._request(
            "GET",
            f"databases/{self.database.database_id}/nodes/{self.id}/connections",
            params=params,
        )
        
        return response
    
    def connect(
        self,
        target_id: str,
        relationship_type: str,
        strength: float = 1.0,
        custom_type: Optional[str] = None,
        bidirectional: bool = False,
    ) -> Dict[str, Any]:
        """
        Create a connection to another node.
        
        This method creates the connection directly via the API.
        
        Args:
            target_id: The ID of the target node.
            relationship_type: The type of relationship.
            strength: The strength of the connection (0.0 to 1.0).
            custom_type: Optional custom relationship type name (for custom types).
            bidirectional: Whether to create a connection in both directions.
            
        Returns:
            The created connection.
            
        Raises:
            ApiError: If the API returns an error response.
        """
        data = {
            "target_id": target_id,
            "type_name": relationship_type,
            "strength": strength,
            "bidirectional": bidirectional,
        }
        
        if custom_type:
            data["custom_type"] = custom_type
            
        response = self.database.client._request(
            "POST",
            f"databases/{self.database.database_id}/nodes/{self.id}/connections",
            json_data=data,
        )
        
        # Also update the local connections list
        connection = {
            "target_id": target_id,
            "type_name": relationship_type,
            "strength": strength,
        }
        
        if custom_type:
            connection["custom_type"] = custom_type
            
        self._connections.append(connection)
        
        return response
    
    def disconnect(
        self,
        target_id: str,
        bidirectional: bool = False,
    ) -> bool:
        """
        Remove a connection to another node.
        
        This method removes the connection directly via the API.
        
        Args:
            target_id: The ID of the target node.
            bidirectional: Whether to remove the connection in both directions.
            
        Returns:
            True if the connection was removed, False if it did not exist.
            
        Raises:
            ApiError: If the API returns an error response.
        """
        params = {}
        if bidirectional:
            params["bidirectional"] = "true"
            
        self.database.client._request(
            "DELETE",
            f"databases/{self.database.database_id}/nodes/{self.id}/connections/{target_id}",
            params=params,
        )
        
        # Also update the local connections list
        initial_length = len(self._connections)
        self._connections = [
            conn for conn in self._connections if conn["target_id"] != target_id
        ]
        
        return len(self._connections) < initial_length