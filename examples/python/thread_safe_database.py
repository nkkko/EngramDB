"""
Example showing how to use the thread-safe database in Python.

This example demonstrates how to:
1. Create a thread-safe database
2. Use it from multiple threads
3. Handle concurrent operations safely
"""

import threading
import time
import uuid
from typing import List

import engramdb

def create_memory_nodes(db: engramdb.ThreadSafeDatabase, count: int) -> List[uuid.UUID]:
    """Create a number of memory nodes in the database"""
    ids = []
    for i in range(count):
        # Create a memory node with a simple embedding
        embedding = [float(i), float(i+1), float(i+2)]
        node = engramdb.MemoryNode(embedding)
        
        # Add some attributes
        node.set_attribute("name", f"Node {i}")
        node.set_attribute("value", i)
        
        # Save to database
        node_id = db.save(node)
        ids.append(node_id)
        
    return ids

def worker_function(db: engramdb.ThreadSafeDatabase, thread_index: int, iterations: int):
    """Worker function that performs database operations"""
    print(f"Thread {thread_index} starting")
    
    # Create some nodes specific to this thread
    thread_nodes = []
    for i in range(iterations):
        # Create a memory node with a unique embedding
        embedding = [float(thread_index * 100 + i), 0.0, 0.0]
        node = engramdb.MemoryNode(embedding)
        
        # Add thread-specific metadata
        node.set_attribute("thread", thread_index)
        node.set_attribute("iteration", i)
        
        # Save to database
        node_id = db.save(node)
        thread_nodes.append(node_id)
        
    # Now load and verify each node
    for i, node_id in enumerate(thread_nodes):
        loaded_node = db.load(node_id)
        
        # Verify the metadata
        assert loaded_node.get_attribute("thread") == thread_index
        assert loaded_node.get_attribute("iteration") == i
        
    # Create some connections between nodes
    if len(thread_nodes) >= 2:
        for i in range(len(thread_nodes) - 1):
            db.connect(
                thread_nodes[i], 
                thread_nodes[i+1], 
                "Sequence", 
                0.9
            )
    
    print(f"Thread {thread_index} completed")

def main():
    # Create a thread-safe in-memory database
    db = engramdb.ThreadSafeDatabase.in_memory_with_hnsw()
    
    # Create some initial data
    initial_ids = create_memory_nodes(db, 10)
    print(f"Created {len(initial_ids)} initial nodes")
    
    # Create multiple threads to operate on the database
    threads = []
    for i in range(5):
        thread = threading.Thread(
            target=worker_function, 
            args=(db, i, 20)
        )
        threads.append(thread)
    
    # Start all threads
    for thread in threads:
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    # Verify the final state of the database
    count = db.len()
    print(f"Database contains {count} memory nodes")
    
    # Get all node IDs
    all_ids = db.list_all()
    print(f"Total nodes: {len(all_ids)}")
    
    # Try a vector search using one of the thread's specific vectors
    query_vector = [200.0, 0.0, 0.0]  # From thread 2, iteration 0
    results = db.search_similar(query_vector, 5, 0.1)
    
    print("\nSearch results:")
    for node_id, similarity in results:
        node = db.load(node_id)
        thread = node.get_attribute("thread") if node.has_attribute("thread") else "N/A"
        iter_val = node.get_attribute("iteration") if node.has_attribute("iteration") else "N/A"
        print(f"  Node {node_id}: similarity={similarity:.4f}, thread={thread}, iteration={iter_val}")

if __name__ == "__main__":
    main()