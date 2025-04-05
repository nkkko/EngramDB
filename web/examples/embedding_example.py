"""
Example demonstrating the use of multilingual-e5-large-instruct embeddings with EngramDB.

This script shows how to:
1. Generate embeddings from text using the model
2. Store these embeddings in EngramDB
3. Perform semantic search using generated embeddings
"""
import os
import sys
import numpy as np
from scipy.spatial.distance import cosine

# Add parent directory to path to import our modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import the embedding utilities
from embedding_utils import generate_embedding_for_memory, generate_embedding_from_query

# Try to import EngramDB, use mock if not available
try:
    from engramdb_py import MemoryNode, Database
    print("Using actual EngramDB implementation")
except ImportError:
    print("EngramDB not available, using mock implementation")
    # Use the same mock implementation from app_full.py
    import uuid
    
    class MockMemoryNode:
        def __init__(self, embeddings=None):
            self.id = str(uuid.uuid4())
            self._embeddings = embeddings or [0.1, 0.2, 0.3, 0.4]
            self._attributes = {}
        
        def get_embeddings(self):
            return np.array(self._embeddings)
        
        def set_embeddings(self, embeddings):
            self._embeddings = embeddings
        
        def get_attribute(self, key):
            return self._attributes.get(key)
        
        def set_attribute(self, key, value):
            self._attributes[key] = value
        
        def attributes(self):
            return self._attributes.items()

    class MockDatabase:
        def __init__(self):
            self.memories = {}
        
        @classmethod
        def in_memory(cls):
            return cls()
        
        def save(self, memory):
            self.memories[memory.id] = memory
            return memory.id
        
        def load(self, memory_id):
            if memory_id not in self.memories:
                raise ValueError(f"Memory with ID {memory_id} not found")
            return self.memories[memory_id]
        
        def list_all(self):
            return list(self.memories.keys())
        
        def search_similar(self, query_vector, limit=10, threshold=0.0):
            # Calculate actual similarity using cosine distance
            results = []
            for memory_id, memory in self.memories.items():
                memory_vector = memory.get_embeddings()
                
                # Skip if embedding dimensions don't match
                if len(memory_vector) != len(query_vector):
                    continue
                
                # Calculate similarity score (1 - cosine distance)
                similarity = 1 - cosine(memory_vector, query_vector)
                
                # Only include if above threshold
                if similarity >= threshold:
                    results.append((memory_id, similarity))
            
            # Sort by similarity (highest first) and return top limit results
            results.sort(key=lambda x: x[1], reverse=True)
            return results[:limit]
            
    MemoryNode = MockMemoryNode
    Database = MockDatabase

def calculate_similarity(vec1, vec2):
    """Calculate cosine similarity between two vectors"""
    return 1 - cosine(vec1, vec2)

def run_example():
    """Run the multilingual embedding example"""
    print("Multilingual E5 Embedding Example")
    print("=================================")
    
    # Create a database
    db = Database.in_memory()
    print("\nCreated in-memory database")
    
    # Sample texts in multiple languages
    memory_texts = [
        ("English", "This article explains how to create embeddings for text in multiple languages."),
        ("Spanish", "Este artículo explica cómo crear incrustaciones para texto en múltiples idiomas."),
        ("French", "Cet article explique comment créer des plongements pour du texte en plusieurs langues."),
        ("German", "Dieser Artikel erklärt, wie man Einbettungen für Text in mehreren Sprachen erstellt."),
    ]
    
    # Create memories from the texts
    print("\nCreating memories with embeddings from different languages:")
    for lang, text in memory_texts:
        # Generate embedding from text
        embedding = generate_embedding_for_memory(text, category=lang)
        
        if embedding is None:
            print(f"  Model not available - using random embedding for {lang}")
            # If model isn't available, create a random embedding with 384 dimensions (e5-large output size)
            embedding = np.random.normal(0, 1, 384)
            embedding = embedding / np.linalg.norm(embedding)
        
        # Create memory with the embedding
        memory = MemoryNode(embedding)
        memory.set_attribute("title", f"{lang} Example")
        memory.set_attribute("content", text)
        memory.set_attribute("language", lang)
        
        # Save to database
        memory_id = db.save(memory)
        print(f"  Created memory: {lang} Example (ID: {memory_id})")
    
    # Example search queries in different languages
    print("\nPerforming semantic searches in different languages:")
    
    search_queries = [
        ("English", "How to create embeddings for multilingual text?"),
        ("Spanish", "¿Cómo crear incrustaciones para textos multilingües?"),
        ("French", "Comment créer des embeddings pour du texte multilingue?"),
    ]
    
    for lang, query in search_queries:
        print(f"\nSearch query ({lang}): {query}")
        
        # Generate embedding for the query
        query_embedding = generate_embedding_from_query(query)
        
        if query_embedding is None:
            print("  Model not available - using random embedding for query")
            # If model isn't available, create a random embedding
            query_embedding = np.random.normal(0, 1, 384)
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # Search for similar memories
        results = db.search_similar(query_embedding, limit=10, threshold=0.0)
        
        print(f"  Found {len(results)} results:")
        for memory_id, similarity in results:
            memory = db.load(memory_id)
            lang = memory.get_attribute("language")
            title = memory.get_attribute("title")
            print(f"  - {title} (similarity: {similarity:.4f})")
            
    print("\nExample completed!")

if __name__ == "__main__":
    run_example()