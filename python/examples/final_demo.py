"""
Basic usage example for the EngramDB Python SDK.
"""
import sys
import os

# Add the parent directory to sys.path so Python can find the engramdb package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from engramdb_py import MemoryNode, Database, RelationshipType
import numpy as np

def run_example():
    try:
        # Create an in-memory database
        print("Creating in-memory database...")
        db = Database.in_memory()
        
        # Create example memories
        print("\nCreating sample memories...")
        
        # Memory 1: Meeting notes
        embeddings1 = np.array([0.1, 0.3, 0.5, 0.1], dtype=np.float32)
        memory1 = MemoryNode(embeddings1)
        memory1.set_attribute("title", "Meeting Notes")
        memory1.set_attribute("category", "work")
        memory1.set_attribute("importance", 0.8)
        memory1_id = db.save(memory1)
        print(f"Created memory: Meeting Notes (ID: {memory1_id})")
        
        # Memory 2: Shopping list
        embeddings2 = np.array([0.8, 0.1, 0.0, 0.1], dtype=np.float32)
        memory2 = MemoryNode(embeddings2)
        memory2.set_attribute("title", "Shopping List")
        memory2.set_attribute("category", "personal")
        memory2.set_attribute("importance", 0.4)
        memory2_id = db.save(memory2)
        print(f"Created memory: Shopping List (ID: {memory2_id})")
        
        # Memory 3: Project idea
        embeddings3 = np.array([0.2, 0.4, 0.4, 0.0], dtype=np.float32)
        memory3 = MemoryNode(embeddings3)
        memory3.set_attribute("title", "Project Idea")
        memory3.set_attribute("category", "work")
        memory3.set_attribute("importance", 0.9)
        memory3_id = db.save(memory3)
        print(f"Created memory: Project Idea (ID: {memory3_id})")
        
        # List all memories
        all_ids = db.list_all()
        print(f"\nDatabase contains {len(all_ids)} memories")
        
        # Vector similarity search
        print("\nPerforming vector similarity search...")
        query_vector = np.array([0.15, 0.35, 0.45, 0.05], dtype=np.float32)
        
        results = db.search_similar(query_vector, limit=5, threshold=0.0)
        print(f"Found {len(results)} similar memories:")
        
        for memory_id, similarity in results:
            memory = db.load(memory_id)
            print(f"  {memory.get_attribute('title')} (similarity: {similarity:.4f})")
        
        print("\nEngramDB Python SDK demo complete!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_example()