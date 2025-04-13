"""
EngramDB Python Module (Mock)

This is a mock implementation of the EngramDB Python module for testing 
the MCP server when the actual module is not installed.
"""

import os
import uuid
import json
from enum import Enum
from typing import Dict, List, Any, Optional

class RelationshipType(Enum):
    """Types of relationships between memory nodes."""
    REFERENCES = 0
    SIMILAR_TO = 1
    BEFORE = 2
    AFTER = 3
    PARENT_OF = 4
    CHILD_OF = 5
    CUSTOM = 6

class MemoryNode:
    """A memory node in EngramDB."""
    
    def __init__(self, id: str, content: str, metadata: Dict[str, Any] = None):
        self.id = id
        self.content = content
        self.metadata = metadata or {}
        self.score = None  # For search results

class Relationship:
    """A relationship between two memory nodes."""
    
    def __init__(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: RelationshipType,
        custom_type: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ):
        self.source_id = source_id
        self.target_id = target_id
        self.relationship_type = relationship_type
        self.custom_type = custom_type
        self.metadata = metadata or {}

class ThreadSafeDatabase:
    """A thread-safe database for EngramDB."""
    
    def __init__(self, path: str):
        self.path = path
        # Create the database directory if it doesn't exist
        os.makedirs(path, exist_ok=True)
        
        # In-memory stores for testing
        self.memories = {}
        self.relationships = []
        
        # Load existing memories if any
        self._load_memories()
        
        print(f"Initialized EngramDB ThreadSafeDatabase at: {path}")
    
    def _load_memories(self):
        """Load memories from disk if they exist."""
        memories_dir = os.path.join(self.path, "memories")
        if os.path.exists(memories_dir):
            for filename in os.listdir(memories_dir):
                if filename.endswith(".json"):
                    with open(os.path.join(memories_dir, filename), "r") as f:
                        data = json.load(f)
                        memory = MemoryNode(
                            id=data["id"],
                            content=data["content"],
                            metadata=data.get("metadata", {})
                        )
                        self.memories[memory.id] = memory
        
        relationships_file = os.path.join(self.path, "relationships.json")
        if os.path.exists(relationships_file):
            with open(relationships_file, "r") as f:
                relationships_data = json.load(f)
                for rel_data in relationships_data:
                    relationship = Relationship(
                        source_id=rel_data["source_id"],
                        target_id=rel_data["target_id"],
                        relationship_type=RelationshipType[rel_data["relationship_type"]],
                        custom_type=rel_data.get("custom_type"),
                        metadata=rel_data.get("metadata", {})
                    )
                    self.relationships.append(relationship)
    
    def _save_memory(self, memory: MemoryNode):
        """Save a memory to disk."""
        memories_dir = os.path.join(self.path, "memories")
        os.makedirs(memories_dir, exist_ok=True)
        
        memory_file = os.path.join(memories_dir, f"{memory.id}.json")
        with open(memory_file, "w") as f:
            json.dump({
                "id": memory.id,
                "content": memory.content,
                "metadata": memory.metadata
            }, f, indent=2)
    
    def _save_relationships(self):
        """Save all relationships to disk."""
        relationships_file = os.path.join(self.path, "relationships.json")
        
        relationships_data = []
        for rel in self.relationships:
            relationships_data.append({
                "source_id": rel.source_id,
                "target_id": rel.target_id,
                "relationship_type": rel.relationship_type.name,
                "custom_type": rel.custom_type,
                "metadata": rel.metadata
            })
        
        with open(relationships_file, "w") as f:
            json.dump(relationships_data, f, indent=2)
    
    def store_memory(self, memory: MemoryNode) -> bool:
        """Store a memory in the database."""
        self.memories[memory.id] = memory
        self._save_memory(memory)
        return True
    
    def get_memory(self, id: str) -> Optional[MemoryNode]:
        """Get a memory by ID."""
        return self.memories.get(id)
    
    def get_all_memories(self) -> List[MemoryNode]:
        """Get all memories in the database."""
        return list(self.memories.values())
    
    def delete_memory(self, id: str) -> bool:
        """Delete a memory by ID."""
        if id in self.memories:
            # Remove the memory
            del self.memories[id]
            
            # Remove file if it exists
            memory_file = os.path.join(self.path, "memories", f"{id}.json")
            if os.path.exists(memory_file):
                os.remove(memory_file)
            
            # Remove any relationships involving this memory
            self.relationships = [
                rel for rel in self.relationships 
                if rel.source_id != id and rel.target_id != id
            ]
            self._save_relationships()
            
            return True
        return False
    
    def search_memories(self, query: str, limit: int = 10) -> List[MemoryNode]:
        """Search memories by content (simple substring match for mock)."""
        results = []
        query = query.lower()
        
        for memory in self.memories.values():
            if query in memory.content.lower():
                # Create a copy with score
                result = MemoryNode(
                    id=memory.id,
                    content=memory.content,
                    metadata=memory.metadata
                )
                # Simple scoring - just count occurrences
                result.score = memory.content.lower().count(query)
                results.append(result)
        
        # Sort by score (higher is better)
        results.sort(key=lambda x: x.score, reverse=True)
        
        return results[:limit]
    
    def create_relationship(
        self, 
        source_id: str, 
        target_id: str, 
        relationship_type: RelationshipType,
        custom_type: Optional[str] = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        """Create a relationship between two memories."""
        # Check that both memories exist
        if source_id not in self.memories or target_id not in self.memories:
            return False
        
        # Create the relationship
        relationship = Relationship(
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
            custom_type=custom_type,
            metadata=metadata or {}
        )
        
        # Add to relationships
        self.relationships.append(relationship)
        self._save_relationships()
        
        return True
    
    def get_outgoing_relationships(
        self, 
        memory_id: str,
        relationship_type: Optional[RelationshipType] = None,
        custom_type: Optional[str] = None
    ) -> List[Relationship]:
        """Get outgoing relationships from a memory."""
        results = []
        
        for rel in self.relationships:
            if rel.source_id == memory_id:
                # Filter by relationship type if specified
                if relationship_type and rel.relationship_type != relationship_type:
                    continue
                
                # Filter by custom type if specified
                if custom_type and rel.custom_type != custom_type:
                    continue
                
                results.append(rel)
        
        return results
    
    def get_incoming_relationships(
        self, 
        memory_id: str,
        relationship_type: Optional[RelationshipType] = None,
        custom_type: Optional[str] = None
    ) -> List[Relationship]:
        """Get incoming relationships to a memory."""
        results = []
        
        for rel in self.relationships:
            if rel.target_id == memory_id:
                # Filter by relationship type if specified
                if relationship_type and rel.relationship_type != relationship_type:
                    continue
                
                # Filter by custom type if specified
                if custom_type and rel.custom_type != custom_type:
                    continue
                
                results.append(rel)
        
        return results