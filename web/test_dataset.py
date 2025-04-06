#!/usr/bin/env python3
"""
Test script for verifying the sample dataset loaders in EngramDB.

This script:
1. Creates a temporary database
2. Loads both minimal and full sample datasets 
3. Examines the nodes and connections
4. Prints a summary of the dataset structure
"""
import os
import sys
import shutil
import tempfile

# Add parent directory to sys.path to find engramdb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import engramdb_py
    from engramdb_py import Database
except ImportError as e:
    print(f"Error importing EngramDB Python bindings: {e}")
    print("Make sure you've installed the Python bindings with 'pip install -e python/'")
    sys.exit(1)

def print_separator(title=None):
    """Print a separator line with optional title."""
    width = 80
    if title:
        print(f"\n{'-' * 10} {title} {'-' * (width - len(title) - 12)}")
    else:
        print(f"\n{'-' * width}")

def print_node_details(db, node_id, indent=""):
    """Print details about a node including its attributes and connections."""
    try:
        node = db.load(node_id)
        
        # Get basic information
        title = node.get_attribute("title") or "Untitled"
        category = node.get_attribute("category") or "Uncategorized"
        content = node.get_attribute("content") or ""
        
        # Print node info
        print(f"{indent}Node: {title}")
        print(f"{indent}  ID: {node_id}")
        print(f"{indent}  Category: {category}")
        
        # Print first line of content
        if content:
            content_preview = content.split('\n')[0]
            if len(content_preview) > 60:
                content_preview = content_preview[:57] + "..."
            print(f"{indent}  Content: {content_preview}")
        
        # Print connections
        connections = db.get_connections(node_id)
        if connections:
            print(f"{indent}  Outgoing Connections:")
            for conn in connections:
                target_node = db.load(conn["target_id"])
                target_title = target_node.get_attribute("title") or "Untitled"
                print(f"{indent}    → {target_title} ({conn['type']}, strength: {conn['strength']})")
        
        incoming = db.get_connected_to(node_id)
        if incoming:
            print(f"{indent}  Incoming Connections:")
            for conn in incoming:
                source_node = db.load(conn["source_id"])
                source_title = source_node.get_attribute("title") or "Untitled"
                print(f"{indent}    ← {source_title} ({conn['type']}, strength: {conn['strength']})")
                
    except Exception as e:
        print(f"{indent}Error processing node {node_id}: {e}")

def analyze_dataset(db):
    """Analyze the dataset and print summary statistics."""
    try:
        # Get all nodes
        node_ids = db.list_all()
        
        # Count nodes by category
        categories = {}
        for node_id in node_ids:
            node = db.load(node_id)
            category = node.get_attribute("category") or "Uncategorized"
            categories[category] = categories.get(category, 0) + 1
        
        # Count connections
        connection_types = {}
        total_connections = 0
        
        for node_id in node_ids:
            connections = db.get_connections(node_id)
            total_connections += len(connections)
            
            for conn in connections:
                conn_type = conn["type"]
                connection_types[conn_type] = connection_types.get(conn_type, 0) + 1
        
        # Print summary
        print_separator("Dataset Summary")
        print(f"Total Nodes: {len(node_ids)}")
        print(f"Total Connections: {total_connections}")
        
        print("\nNodes by Category:")
        for category, count in sorted(categories.items(), key=lambda x: x[1], reverse=True):
            print(f"  {category}: {count}")
        
        print("\nConnection Types:")
        for conn_type, count in sorted(connection_types.items(), key=lambda x: x[1], reverse=True):
            print(f"  {conn_type}: {count}")
            
    except Exception as e:
        print(f"Error analyzing dataset: {e}")

def main():
    # Create a temporary database
    temp_dir = tempfile.mkdtemp(prefix="engramdb_test_")
    db_path = os.path.join(temp_dir, "test_dataset")
    
    try:
        print(f"Creating temporary database at {db_path}")
        db = Database.file_based(db_path)
        
        # Test both dataset loaders
        print("\nTesting minimal dataset loader...")
        try:
            node_ids = engramdb_py.load_minimal_dataset(db)
            print(f"Success! Loaded {len(node_ids)} nodes from minimal dataset")
            
            # Display the nodes from minimal dataset
            print("\nMinimal dataset nodes:")
            for node_id in node_ids:
                print_node_details(db, node_id, indent="  ")
            
            # Analyze dataset structure
            analyze_dataset(db)
            
        except Exception as e:
            print(f"Error loading minimal dataset: {e}")
        
        # Now test the full dataset
        print("\nTesting full sample dataset loader...")
        try:
            db.clear_all()
            node_ids = engramdb_py.load_sample_dataset(db)
            print(f"Success! Loaded {len(node_ids)} nodes from full dataset")
            
            # Display a sample of nodes from the full dataset
            print("\nSample nodes from full dataset:")
            for node_id in node_ids[:3]:  # Just show the first 3
                print_node_details(db, node_id, indent="  ")
            
            if len(node_ids) > 3:
                print(f"\n  ... and {len(node_ids) - 3} more nodes")
            
            # Analyze dataset structure
            analyze_dataset(db)
            
        except Exception as e:
            print(f"Error loading full dataset: {e}")
    
    finally:
        # Clean up
        print("\nCleaning up temporary files...")
        try:
            shutil.rmtree(temp_dir)
            print(f"Removed temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to remove temporary directory: {e}")

if __name__ == "__main__":
    main()