import pytest
import rtamp
import numpy as np

def test_sum_as_string():
    result = rtamp.sum_as_string(1, 2)
    assert result == "3"

def test_memory_node_creation():
    # Create a memory node with embeddings
    embeddings = [0.1, 0.2, 0.3, 0.4]
    node = rtamp.MemoryNode(embeddings)
    
    # Check that the node has a valid UUID
    assert isinstance(node.id, str)
    assert len(node.id) > 0
    
    # Check that the embeddings are correct
    assert list(node.embeddings) == embeddings
    
    # Test attribute setting and getting
    node.set_attribute("key1", "value1")
    node.set_attribute("key2", 42)
    node.set_attribute("key3", 3.14)
    node.set_attribute("key4", True)
    
    assert node.get_attribute("key1") == "value1"
    assert node.get_attribute("key2") == 42
    assert node.get_attribute("key3") == pytest.approx(3.14)
    assert node.get_attribute("key4") is True
    assert node.get_attribute("non_existent") is None
    
    # Check creation timestamp
    assert node.creation_timestamp > 0

def test_in_memory_database():
    # Create an in-memory database
    db = rtamp.Database.in_memory()
    
    # Create a memory node
    node = rtamp.MemoryNode([0.1, 0.2, 0.3])
    node.set_attribute("name", "Test Node")
    
    # Save the node to the database
    memory_id = db.save(node)
    assert isinstance(memory_id, str)
    
    # Get the list of all nodes
    all_ids = db.list_all()
    assert len(all_ids) == 1
    assert all_ids[0] == memory_id
    
    # Load the node back
    loaded_node = db.load(memory_id)
    assert loaded_node.id == memory_id
    assert loaded_node.get_attribute("name") == "Test Node"
    
    # Delete the node
    db.delete(memory_id)
    all_ids = db.list_all()
    assert len(all_ids) == 0

def test_relationship_types():
    # Test the various relationship types
    causation = rtamp.RelationshipType.causal()
    hierarchical = rtamp.RelationshipType.hierarchical()
    contains = rtamp.RelationshipType.contains()
    sequential = rtamp.RelationshipType.sequential()
    association = rtamp.RelationshipType.association()
    custom = rtamp.RelationshipType.custom("myrelation")
    
    # We can't directly test the enum values, but at least we can check they don't raise exceptions
    assert causation is not None
    assert hierarchical is not None
    assert contains is not None
    assert sequential is not None
    assert association is not None
    assert custom is not None

def test_attribute_filter():
    # Test attribute filters - these are just basic types without much functionality yet
    filter1 = rtamp.AttributeFilter.equals("category", "work")
    filter2 = rtamp.AttributeFilter.greater_than("importance", "0.5")
    filter3 = rtamp.AttributeFilter.less_than("priority", "3")
    
    assert filter1 is not None
    assert filter2 is not None
    assert filter3 is not None