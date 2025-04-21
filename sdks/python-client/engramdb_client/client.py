"""
EngramDB client for connecting to the EngramDB REST API.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, cast

import httpx
import numpy as np
from pydantic import ValidationError

from .database import EngramDatabase
from .exceptions import ApiError, AuthenticationError, ConnectionError, EngramClientError, TimeoutError
from .models import DatabaseConfig, DatabaseInfo

logger = logging.getLogger(__name__)


class EngramClient:
    """
    Client for interacting with the EngramDB REST API.
    
    This class provides methods for managing databases and connections to the API.
    """
    
    def __init__(
        self,
        api_url: str = "http://localhost:8000/v1",
        api_key: Optional[str] = None,
        jwt_token: Optional[str] = None,
        timeout: float = 10.0,
    ):
        """
        Initialize the EngramDB client.
        
        Args:
            api_url: The base URL of the EngramDB API.
            api_key: Optional API key for authentication.
            jwt_token: Optional JWT token for authentication.
            timeout: Request timeout in seconds.
        """
        self.api_url = api_url.rstrip("/")
        self.api_key = api_key
        self.jwt_token = jwt_token
        self.timeout = timeout
        self.client = httpx.Client(timeout=timeout)
        
        # Set up authentication headers
        self.headers: Dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key
        if jwt_token:
            self.headers["Authorization"] = f"Bearer {jwt_token}"
    
    def _handle_response(self, response: httpx.Response) -> Dict[str, Any]:
        """
        Handle the API response and raise appropriate exceptions if needed.
        
        Args:
            response: The HTTP response from the API.
            
        Returns:
            The parsed JSON response.
            
        Raises:
            ApiError: If the API returns an error response.
            AuthenticationError: If authentication fails.
        """
        try:
            if response.status_code == 401:
                raise AuthenticationError()
                
            if response.status_code == 204:
                return {}  # No content
                
            data = response.json()
            
            if response.status_code >= 400:
                raise ApiError(
                    message=data.get("message", "Unknown API error"),
                    status_code=response.status_code,
                    code=data.get("code"),
                    details=data.get("details"),
                )
                
            return data
        except json.JSONDecodeError:
            # If the response is not valid JSON, return the raw text
            if response.status_code >= 400:
                raise ApiError(
                    message=response.text,
                    status_code=response.status_code,
                )
            return {"text": response.text}
    
    def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Make a request to the API.
        
        Args:
            method: The HTTP method to use.
            path: The API endpoint path.
            params: Optional query parameters.
            json_data: Optional JSON request body.
            
        Returns:
            The parsed JSON response.
            
        Raises:
            ConnectionError: If a connection to the API server cannot be established.
            TimeoutError: If the request times out.
            ApiError: If the API returns an error response.
        """
        url = f"{self.api_url}/{path.lstrip('/')}"
        
        try:
            response = self.client.request(
                method=method,
                url=url,
                params=params,
                json=json_data,
                headers=self.headers,
                timeout=self.timeout,
            )
            return self._handle_response(response)
        except httpx.ConnectError as e:
            raise ConnectionError(f"Failed to connect to the API server: {str(e)}")
        except httpx.TimeoutException as e:
            raise TimeoutError(f"Request timed out: {str(e)}")
        except httpx.RequestError as e:
            raise EngramClientError(f"Request error: {str(e)}")
    
    def list_databases(self) -> List[DatabaseInfo]:
        """
        List all databases.
        
        Returns:
            A list of database information objects.
        """
        response = self._request("GET", "databases")
        databases = response.get("databases", [])
        return [DatabaseInfo(**db) for db in databases]
    
    def create_database(
        self,
        name: str,
        config: Optional[DatabaseConfig] = None,
    ) -> EngramDatabase:
        """
        Create a new database.
        
        Args:
            name: The name of the database.
            config: Optional configuration for the database.
            
        Returns:
            A database client for the newly created database.
        """
        data = {"name": name}
        if config:
            data["config"] = config.model_dump()
            
        response = self._request("POST", "databases", json_data=data)
        db_info = DatabaseInfo(**response)
        
        return EngramDatabase(client=self, database_id=db_info.id, name=db_info.name)
    
    def get_database(self, database_id: str) -> EngramDatabase:
        """
        Get a database client by ID.
        
        Args:
            database_id: The ID of the database.
            
        Returns:
            A database client for the specified database.
            
        Raises:
            ApiError: If the database does not exist.
        """
        response = self._request("GET", f"databases/{database_id}")
        db_info = DatabaseInfo(**response)
        
        return EngramDatabase(client=self, database_id=db_info.id, name=db_info.name)
    
    def delete_database(self, database_id: str) -> bool:
        """
        Delete a database.
        
        Args:
            database_id: The ID of the database to delete.
            
        Returns:
            True if the database was deleted, False otherwise.
        """
        self._request("DELETE", f"databases/{database_id}")
        return True
    
    def get_models(self) -> List[Dict[str, Any]]:
        """
        Get a list of available embedding models.
        
        Returns:
            A list of embedding model information.
        """
        response = self._request("GET", "models")
        return response.get("models", [])
    
    def generate_embedding(
        self,
        content: str,
        model: str = "default",
    ) -> np.ndarray:
        """
        Generate embeddings for text content.
        
        Args:
            content: The text content to embed.
            model: The embedding model to use.
            
        Returns:
            A numpy array containing the embedding vector.
        """
        response = self._request(
            "POST",
            "generate_embedding",
            json_data={"content": content, "model": model},
        )
        
        vector = response.get("vector", [])
        return np.array(vector, dtype=np.float32)
    
    def close(self) -> None:
        """Close the HTTP client connection."""
        self.client.close()