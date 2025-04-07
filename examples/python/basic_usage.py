import uuid
import os
import shutil
from typing import List, Tuple
import engramdb_py as engramdb

def main():
    # Initialize storage in a temporary directory
    storage_dir = "./tmp_memories"
    os.makedirs(storage_dir, exist_ok=True)
    db = engramdb.Database.file_based(storage_dir)
    print(f"Storage initialized at: {storage_dir}")
    
    # Create example memories
    create_sample_memories(db)
    
    # List all memories
    memory_ids = db.list_all()
    print(f"Created {len(memory_ids)} memories")
    
    # Query for memories by vector similarity
    print("\nSearching for memories similar to 'meeting notes'...")
    query_vector = [0.1, 0.3, 0.5, 0.2]
    search_results = db.search_similar(query_vector, limit=3, threshold=0.5)
    
    print(f"Found {len(search_results)} similar memories:")
    for memory_id, similarity in search_results:
        node = db.load(memory_id)
        print_memory(node, similarity)
    
    # Clean up
    print("\nCleaning up...")
    for memory_id in memory_ids:
        db.delete(memory_id)
    
    shutil.rmtree(storage_dir)
    print("Done!")

def create_sample_memories(db: engramdb.Database):
    # Memory 1: Meeting notes
    node1 = engramdb.MemoryNode([0.1, 0.3, 0.5, 0.2])
    node1.set_attribute("title", "Meeting Notes")
    node1.set_attribute("importance", 0.8)
    node1.set_attribute("category", "work")
    db.save(node1)
    
    # Memory 2: Shopping list
    node2 = engramdb.MemoryNode([0.8, 0.1, 0.0, 0.3])
    node2.set_attribute("title", "Shopping List")
    node2.set_attribute("importance", 0.3)
    node2.set_attribute("category", "personal")
    db.save(node2)
    
    # Memory 3: Project idea
    node3 = engramdb.MemoryNode([0.2, 0.4, 0.4, 0.1])
    node3.set_attribute("title", "Project Idea")
    node3.set_attribute("importance", 0.9)
    node3.set_attribute("category", "work")
    db.save(node3)

def print_memory(node: engramdb.MemoryNode, similarity: float):
    print(f"  Memory: {node.get_attribute('title')} (similarity: {similarity:.4f})")
    print(f"    Importance: {node.get_attribute('importance')}")
    print(f"    Category: {node.get_attribute('category')}")

if __name__ == "__main__":
    main()