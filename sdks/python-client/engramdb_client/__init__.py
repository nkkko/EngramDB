"""
EngramDB Client - Python SDK for EngramDB REST API

This module provides a client library for interacting with the EngramDB REST API.
"""

__version__ = "0.1.0"

from .client import EngramClient
from .database import EngramDatabase
from .memory_node import MemoryNode
from .models import AttributeFilter, Connection, SearchQuery
from .exceptions import EngramClientError, ApiError, AuthenticationError

__all__ = [
    "EngramClient",
    "EngramDatabase",
    "MemoryNode",
    "AttributeFilter", 
    "Connection",
    "SearchQuery",
    "EngramClientError",
    "ApiError",
    "AuthenticationError",
]