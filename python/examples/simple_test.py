"""
Simple test to verify that the Python binding works.
"""
from engramdb_py import sum_as_string, MemoryNode, Database

def test_sum():
    result = sum_as_string(1, 2)
    print(f"1 + 2 = {result}")
    
def test_memory_node():
    try:
        node = MemoryNode([0.1, 0.2, 0.3])
        print(f"Created memory node with ID: {node.id}")
        
        node.set_attribute("name", "Test Node")
        node.set_attribute("value", 42)
        
        print(f"Node name: {node.get_attribute('name')}")
        print(f"Node value: {node.get_attribute('value')}")
        
        print("Memory node test passed!")
    except Exception as e:
        print(f"Memory node test failed: {e}")

def test_database():
    try:
        db = Database.in_memory()
        node = MemoryNode([0.1, 0.2, 0.3])
        node.set_attribute("name", "DB Test Node")
        
        node_id = db.save(node)
        print(f"Saved node with ID: {node_id}")
        
        loaded = db.load(node_id)
        print(f"Loaded node name: {loaded.get_attribute('name')}")
        
        print("Database test passed!")
    except Exception as e:
        print(f"Database test failed: {e}")

def main():
    print("=== Testing sum_as_string ===")
    test_sum()
    
    print("\n=== Testing MemoryNode ===")
    test_memory_node()
    
    print("\n=== Testing Database ===")
    test_database()
    
    print("\nAll tests complete!")

if __name__ == "__main__":
    main()