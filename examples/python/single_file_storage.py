import os
import shutil
from typing import List, Any
import engramdb_py as engramdb

def main():
    # Create a file-based database
    db_file = "./single_file_db.engramdb"
    db = engramdb.Database.file_based(db_file)
    print(f"Database created at: {db_file}")
    
    # Store sample memories
    memory_ids = store_sample_memories(db)
    
    # Query and display memories
    query_and_display(db)
    
    # Delete memories and clean up
    print("\nCleaning up...")
    for memory_id in memory_ids:
        db.delete(memory_id)
    
    # Remove the database file
    if os.path.exists(db_file):
        os.remove(db_file)
    print("Done!")

def store_sample_memories(db: engramdb.Database) -> List[Any]:
    """Store several sample memories in the database"""
    memory_ids = []
    
    # Memory 1: Technical note
    memory1 = engramdb.MemoryNode([0.1, 0.2, 0.3, 0.1])
    memory1.set_attribute("title", "Database Optimization")
    memory1.set_attribute("category", "technical")
    memory1.set_attribute("content", "Notes on improving database query performance")
    memory_id1 = db.save(memory1)
    memory_ids.append(memory_id1)
    
    # Memory 2: Personal note
    memory2 = engramdb.MemoryNode([0.8, 0.1, 0.1, 0.9])
    memory2.set_attribute("title", "Vacation Ideas")
    memory2.set_attribute("category", "personal")
    memory2.set_attribute("content", "List of potential vacation destinations for next year")
    memory_id2 = db.save(memory2)
    memory_ids.append(memory_id2)
    
    # Memory 3: Work meeting
    memory3 = engramdb.MemoryNode([0.3, 0.7, 0.2, 0.1])
    memory3.set_attribute("title", "Team Coordination")
    memory3.set_attribute("category", "work")
    memory3.set_attribute("content", "Meeting notes from weekly team sync")
    memory_id3 = db.save(memory3)
    memory_ids.append(memory_id3)
    
    # Memory 4: Creative idea
    memory4 = engramdb.MemoryNode([0.5, 0.5, 0.6, 0.7])
    memory4.set_attribute("title", "New Project Concept")
    memory4.set_attribute("category", "creative")
    memory4.set_attribute("content", "Initial thoughts on a new side project")
    memory_id4 = db.save(memory4)
    memory_ids.append(memory_id4)
    
    print(f"Stored {len(memory_ids)} memories in the database")
    return memory_ids

def query_and_display(db: engramdb.Database):
    """Run various queries on the database and display results"""
    # List all memories
    all_ids = db.list_all()
    print(f"\nDatabase contains {len(all_ids)} memories")
    
    # Display all memories
    print("\nAll memories:")
    for memory_id in all_ids:
        memory = db.load(memory_id)
        title = memory.get_attribute("title")
        category = memory.get_attribute("category")
        print(f"  - {title} (Category: {category})")
    
    # Search by vector similarity for creative ideas
    print("\nMemories similar to creative thinking:")
    creative_vector = [0.6, 0.5, 0.6, 0.7]  # Vector representing creative thinking
    creative_results = db.search_similar(creative_vector, limit=2, threshold=0.0)
    
    for memory_id, similarity in creative_results:
        memory = db.load(memory_id)
        title = memory.get_attribute("title")
        content = memory.get_attribute("content")
        print(f"  - {title} (similarity: {similarity:.4f})")
        print(f"    Content: {content}")
    
    # Search by vector similarity for work-related content
    print("\nMemories similar to work-related content:")
    work_vector = [0.3, 0.7, 0.2, 0.2]  # Vector representing work-related content
    work_results = db.search_similar(work_vector, limit=2, threshold=0.0)
    
    for memory_id, similarity in work_results:
        memory = db.load(memory_id)
        title = memory.get_attribute("title")
        content = memory.get_attribute("content")
        print(f"  - {title} (similarity: {similarity:.4f})")
        print(f"    Content: {content}")

if __name__ == "__main__":
    main()