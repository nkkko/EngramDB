"""
Simple test to verify that the EngramDB Python binding works.
Demonstrates the basic operations for agent memory management.
"""
from engramdb_py import MemoryNode, Database
import numpy as np

def test_memory_node():
    try:
        # Create a memory node representing a bug finding from an AI coding agent
        vector = np.array([0.7, 0.2, 0.3, 0.1], dtype=np.float32)
        node = MemoryNode(vector)
        
        # Set attributes for the memory
        node.set_attribute("memory_type", "bug_report")
        node.set_attribute("title", "Null Reference Bug")
        node.set_attribute("severity", 0.8)
        
        print(f"Created memory node with ID: {node.id}")
        print(f"Memory type: {node.get_attribute('memory_type')}")
        print(f"Title: {node.get_attribute('title')}")
        print(f"Severity: {node.get_attribute('severity')}")
        
        print("Memory node test passed!")
    except Exception as e:
        print(f"Memory node test failed: {e}")

def test_database():
    try:
        # Create an in-memory database
        db = Database.in_memory()
        
        # Create two related memories: a bug and its fix
        bug = MemoryNode(np.array([0.7, 0.2, 0.3, 0.1], dtype=np.float32))
        bug.set_attribute("memory_type", "bug_report")
        bug.set_attribute("title", "Memory Leak in UserController")
        
        fix = MemoryNode(np.array([0.6, 0.3, 0.3, 0.2], dtype=np.float32))
        fix.set_attribute("memory_type", "code_fix")
        fix.set_attribute("title", "Resource Disposal Implementation")
        
        # Save both memories
        bug_id = db.save(bug)
        fix_id = db.save(fix)
        
        print(f"Saved bug report with ID: {bug_id}")
        print(f"Saved code fix with ID: {fix_id}")
        
        # Connect the bug to its fix
        bug = db.load(bug_id)
        bug.add_connection(fix_id, "fixed_by")
        db.save(bug)
        
        # Verify the connection
        reloaded_bug = db.load(bug_id)
        connections = reloaded_bug.get_connections()
        
        if connections and connections[0][0] == fix_id:
            print(f"Connection verified: Bug is fixed by memory {fix_id}")
            
            # Load the connected fix
            connected_fix = db.load(connections[0][0])
            print(f"Connected fix title: {connected_fix.get_attribute('title')}")
            
            print("Database connection test passed!")
        else:
            print("Failed to verify connection")
            
    except Exception as e:
        print(f"Database test failed: {e}")

def main():
    print("=== Testing EngramDB for AI Coding Agent ===")
    
    print("\n--- Testing MemoryNode ---")
    test_memory_node()
    
    print("\n--- Testing Database with Connections ---")
    test_database()
    
    print("\nAll tests complete!")

if __name__ == "__main__":
    main()