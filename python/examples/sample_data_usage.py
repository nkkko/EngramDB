"""
Example script demonstrating how to use the sample datasets in EngramDB.

This script shows how to:
1. Create a new database instance
2. Load a sample dataset
3. Explore the nodes and connections
"""
import os
import sys
import engramdb_py

def main():
    # Create a file-based database
    db_path = "./sample_data_example.db"
    
    print(f"Creating database at {db_path}...")
    db = engramdb_py.Database.file_based(db_path)
    
    # Load the minimal sample dataset
    print("Loading minimal sample dataset...")
    try:
        node_ids = engramdb_py.load_minimal_dataset(db)
        print(f"Loaded {len(node_ids)} nodes into the database")
    except Exception as e:
        print(f"Error loading minimal dataset: {e}")
        sys.exit(1)
    
    # Display all nodes in the database
    print("\nListing all memory nodes:")
    for node_id in db.list_all():
        node = db.load(node_id)
        title = node.get_attribute("title") or "Untitled"
        category = node.get_attribute("category") or "Uncategorized"
        
        print(f"  Node {node_id}: {title} ({category})")
        
        # List its connections
        connections = db.get_connections(node_id)
        if connections:
            print("    Connections:")
            for conn in connections:
                target_node = db.load(conn["target_id"])
                target_title = target_node.get_attribute("title") or "Untitled"
                print(f"      -> {target_title} ({conn['type']}, strength: {conn['strength']})")
    
    # Perform a vector search
    print("\nPerforming vector search for 'requirements'...")
    search_results = db.search_similar([0.9, 0.1, 0.2], 3, 0.0)
    print("Search results:")
    for node_id, similarity in search_results:
        node = db.load(node_id)
        title = node.get_attribute("title") or "Untitled"
        print(f"  {title}: similarity {similarity:.2f}")
    
    # Load the full AI bugfixing dataset (optional)
    load_full = os.environ.get("LOAD_FULL_DATASET", "0").lower() in ("1", "true", "yes")
    if load_full:
        print("\nLoading full AI bugfix workflow dataset...")
        try:
            db.clear_all()
            node_ids = engramdb_py.load_sample_dataset(db)
            print(f"Loaded {len(node_ids)} nodes from the full dataset")
            
            # Display a few nodes as examples
            print("\nSample nodes from the full dataset:")
            for i, node_id in enumerate(node_ids[:3]):
                node = db.load(node_id)
                title = node.get_attribute("title") or "Untitled"
                category = node.get_attribute("category") or "Uncategorized"
                print(f"  Node {i+1}: {title} ({category})")
            
            if len(node_ids) > 3:
                print(f"  ... and {len(node_ids) - 3} more nodes")
        except Exception as e:
            print(f"Error loading full dataset: {e}")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()