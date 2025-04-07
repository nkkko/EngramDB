import os
import shutil
from typing import List, Dict, Any
import engramdb_py as engramdb

def main():
    # Create a hybrid memory system with both in-memory and file-based storage
    memory_db = create_memory_database()
    file_db = create_file_database("./file_db")
    
    # Create and store memories in both systems
    memory_ids = store_sample_memories(memory_db)
    file_ids = store_sample_memories(file_db)
    
    # Query both databases
    print("\nIn-Memory Database:")
    query_database(memory_db)
    
    print("\nFile-Based Database:")
    query_database(file_db)
    
    # Clean up file database
    print("\nCleaning up file database...")
    for memory_id in file_ids:
        file_db.delete(memory_id)
    
    shutil.rmtree("./file_db")
    print("Done!")

def create_memory_database() -> engramdb.Database:
    """Create an in-memory database"""
    db = engramdb.Database.in_memory()
    print("In-memory database created")
    return db

def create_file_database(path: str) -> engramdb.Database:
    """Create a file-based database"""
    os.makedirs(path, exist_ok=True)
    db = engramdb.Database.file_based(path)
    print(f"File-based database created at: {path}")
    return db

def store_sample_memories(db: engramdb.Database) -> List[Any]:
    """Store sample memories in the provided database"""
    # First memory
    node1 = engramdb.MemoryNode([0.1, 0.3, 0.5, 0.7])
    node1.set_attribute("title", "Meeting Notes")
    node1.set_attribute("content", "Discussion about new project timeline")
    memory_id1 = db.save(node1)
    
    # Second memory
    node2 = engramdb.MemoryNode([0.8, 0.2, 0.1, 0.3])
    node2.set_attribute("title", "Research Findings")
    node2.set_attribute("content", "Key insights from literature review")
    memory_id2 = db.save(node2)
    
    print(f"Stored 2 memories in the database")
    return [memory_id1, memory_id2]

def query_database(db: engramdb.Database):
    """Run sample queries on the database"""
    # List all memories
    all_ids = db.list_all()
    print(f"Database contains {len(all_ids)} memories")
    
    # Search by vector similarity
    query_vector = [0.2, 0.3, 0.4, 0.5]
    results = db.search_similar(query_vector, limit=2, threshold=0.0)
    
    print(f"Found {len(results)} memories similar to query vector:")
    for memory_id, similarity in results:
        memory = db.load(memory_id)
        title = memory.get_attribute("title")
        content = memory.get_attribute("content")
        print(f"  - {title} (similarity: {similarity:.4f})")
        print(f"    Content: {content}")

if __name__ == "__main__":
    main()