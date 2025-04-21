"""
Data models for the EngramDB client.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Literal

import numpy as np
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """Configuration for an EngramDB database."""
    
    storage_type: str = "MultiFile"
    storage_path: Optional[str] = None
    cache_size: int = 100
    vector_algorithm: str = "HNSW"
    hnsw_config: Optional[Dict[str, Any]] = None


class DatabaseInfo(BaseModel):
    """Information about an EngramDB database."""
    
    id: str
    name: str
    storage_type: str
    node_count: int
    created_at: datetime
    config: Optional[DatabaseConfig] = None


class Connection(BaseModel):
    """A connection between memory nodes."""
    
    target_id: str
    type_name: str
    strength: float = 1.0
    custom_type: Optional[str] = None
    

class ConnectionInfo(BaseModel):
    """Detailed information about a connection."""
    
    source_id: str
    target_id: str
    type_name: str
    strength: float
    custom_type: Optional[str] = None


class RelationshipType(str, Enum):
    """Types of relationships between memory nodes."""
    
    ASSOCIATION = "Association"
    CAUSATION = "Causation"
    SEQUENCE = "Sequence"
    CONTAINS = "Contains"
    PART_OF = "PartOf"
    REFERENCE = "Reference"
    CUSTOM = "Custom"


class FilterOperation(str, Enum):
    """Operations for attribute filters."""
    
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    GREATER_OR_EQUAL = "greater_or_equal"
    LESS_OR_EQUAL = "less_or_equal"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    EXISTS = "exists"


class AttributeFilter(BaseModel):
    """A filter for querying memory nodes by attributes."""
    
    field: str
    operation: FilterOperation
    value: Optional[Any] = None
    
    @classmethod
    def equals(cls, field: str, value: Any) -> "AttributeFilter":
        """Create an equals filter."""
        return cls(field=field, operation=FilterOperation.EQUALS, value=value)
    
    @classmethod
    def not_equals(cls, field: str, value: Any) -> "AttributeFilter":
        """Create a not equals filter."""
        return cls(field=field, operation=FilterOperation.NOT_EQUALS, value=value)
    
    @classmethod
    def greater_than(cls, field: str, value: Union[int, float]) -> "AttributeFilter":
        """Create a greater than filter."""
        return cls(field=field, operation=FilterOperation.GREATER_THAN, value=value)
    
    @classmethod
    def less_than(cls, field: str, value: Union[int, float]) -> "AttributeFilter":
        """Create a less than filter."""
        return cls(field=field, operation=FilterOperation.LESS_THAN, value=value)
    
    @classmethod
    def greater_or_equal(cls, field: str, value: Union[int, float]) -> "AttributeFilter":
        """Create a greater than or equal filter."""
        return cls(field=field, operation=FilterOperation.GREATER_OR_EQUAL, value=value)
    
    @classmethod
    def less_or_equal(cls, field: str, value: Union[int, float]) -> "AttributeFilter":
        """Create a less than or equal filter."""
        return cls(field=field, operation=FilterOperation.LESS_OR_EQUAL, value=value)
    
    @classmethod
    def contains(cls, field: str, value: str) -> "AttributeFilter":
        """Create a contains filter."""
        return cls(field=field, operation=FilterOperation.CONTAINS, value=value)
    
    @classmethod
    def starts_with(cls, field: str, value: str) -> "AttributeFilter":
        """Create a starts with filter."""
        return cls(field=field, operation=FilterOperation.STARTS_WITH, value=value)
    
    @classmethod
    def ends_with(cls, field: str, value: str) -> "AttributeFilter":
        """Create an ends with filter."""
        return cls(field=field, operation=FilterOperation.ENDS_WITH, value=value)
    
    @classmethod
    def exists(cls, field: str) -> "AttributeFilter":
        """Create an exists filter."""
        return cls(field=field, operation=FilterOperation.EXISTS)


class SearchQuery(BaseModel):
    """A query for searching memory nodes."""
    
    vector: Optional[List[float]] = None
    content: Optional[str] = None
    model: str = "default"
    filters: List[AttributeFilter] = Field(default_factory=list)
    limit: int = 10
    threshold: float = 0.0
    include_vectors: bool = False
    include_connections: bool = False


class MemoryNodeData(BaseModel):
    """Data for a memory node."""
    
    id: str
    vector: Optional[List[float]] = None
    attributes: Dict[str, Any] = Field(default_factory=dict)
    connections: Optional[List[ConnectionInfo]] = None
    created_at: datetime
    updated_at: datetime
    content: Optional[str] = None


class SearchResult(BaseModel):
    """A search result containing a memory node and its similarity score."""
    
    node: MemoryNodeData
    similarity: float