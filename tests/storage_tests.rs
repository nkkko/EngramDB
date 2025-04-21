use engramdb::{
    core::{AttributeValue, MemoryNode},
    database::Database,
    Result,
};
use std::collections::HashSet;
use tempfile::tempdir;
use uuid::Uuid;

/// Test scenario: Basic persistence
///
/// This test verifies that data saved to a file-based database
/// persists after closing and reopening the database.
#[test]
fn test_storage_basic_persistence() -> Result<()> {
    // Create a temporary directory for our test
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test_persistence.db");

    // Generate some test data
    let num_nodes = 10;
    let mut node_ids = Vec::with_capacity(num_nodes);

    // First, create and populate the database
    {
        let db = Database::file_based(&db_path)?;

        for i in 0..num_nodes {
            let mut node = MemoryNode::new(vec![i as f32, (i * 2) as f32, (i * 3) as f32]);
            node.set_attribute("index".to_string(), AttributeValue::Integer(i as i64));
            node.set_attribute(
                "name".to_string(),
                AttributeValue::String(format!("Node {}", i)),
            );
            let id = db.save(&node)?;
            node_ids.push(id);
        }

        // Verify database has the correct number of nodes
        assert_eq!(db.len()?, num_nodes);
    }
    // Database is closed when it goes out of scope

    // Now, reopen the database and verify data persisted
    {
        let db = Database::file_based(&db_path)?;

        // Verify database has the correct number of nodes
        assert_eq!(db.len()?, num_nodes);

        // Verify each node was correctly saved and can be loaded
        for (i, id) in node_ids.iter().enumerate() {
            let node = db.load(*id)?;

            // Check attributes
            assert_eq!(
                node.get_attribute("index").unwrap().to_string(),
                i.to_string()
            );
            assert_eq!(
                node.get_attribute("name").unwrap().to_string(),
                format!("\"Node {}\"", i)
            );

            // Check embedding
            let embedding = node.embedding();
            assert_eq!(embedding.len(), 3);
            assert_eq!(embedding[0], i as f32);
            assert_eq!(embedding[1], (i * 2) as f32);
            assert_eq!(embedding[2], (i * 3) as f32);
        }
    }

    Ok(())
}

/// Test scenario: Database updates
///
/// This test verifies that updates to existing nodes are properly
/// persisted to storage.
#[test]
fn test_storage_database_updates() -> Result<()> {
    // Create a temporary directory for our test
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test_updates.db");

    // Generate a node to test with
    let node_id;

    // Create initial node
    {
        let db = Database::file_based(&db_path)?;

        let mut node = MemoryNode::new(vec![1.0, 2.0, 3.0]);
        node.set_attribute("version".to_string(), AttributeValue::Integer(1));
        node_id = db.save(&node)?;
    }

    // Update the node
    {
        let db = Database::file_based(&db_path)?;

        let mut node = db.load(node_id)?;
        assert_eq!(node.get_attribute("version").unwrap().to_string(), "1");

        // Update the node
        node.set_attribute("version".to_string(), AttributeValue::Integer(2));
        node.set_attribute(
            "new_attr".to_string(),
            AttributeValue::String("added".to_string()),
        );
        db.save(&node)?;
    }

    // Verify the update persisted
    {
        let db = Database::file_based(&db_path)?;

        let node = db.load(node_id)?;
        assert_eq!(node.get_attribute("version").unwrap().to_string(), "2");
        assert_eq!(
            node.get_attribute("new_attr").unwrap().to_string(),
            "\"added\""
        );
    }

    Ok(())
}

/// Test scenario: Delete operations
///
/// This test verifies that node deletion is properly handled
/// and persisted to storage.
#[test]
fn test_storage_delete_operations() -> Result<()> {
    // Create a temporary directory for our test
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test_delete.db");

    // Create nodes
    let node_ids;
    {
        let db = Database::file_based(&db_path)?;

        // Create 5 nodes
        let mut ids = Vec::with_capacity(5);
        for i in 0..5 {
            let mut node = MemoryNode::new(vec![i as f32]);
            node.set_attribute("index".to_string(), AttributeValue::Integer(i as i64));
            ids.push(db.save(&node)?);
        }
        node_ids = ids;

        // Verify database has 5 nodes
        assert_eq!(db.len()?, 5);
    }

    // Delete some nodes
    {
        let db = Database::file_based(&db_path)?;

        // Delete nodes 1 and 3
        db.delete(node_ids[1])?;
        db.delete(node_ids[3])?;

        // Verify database now has 3 nodes
        assert_eq!(db.len()?, 3);
    }

    // Verify deletions persisted
    {
        let db = Database::file_based(&db_path)?;

        // Verify database still has 3 nodes
        assert_eq!(db.len()?, 3);

        // Verify nodes 0, 2, 4 exist
        let node0 = db.load(node_ids[0]);
        let node1 = db.load(node_ids[1]);
        let node2 = db.load(node_ids[2]);
        let node3 = db.load(node_ids[3]);
        let node4 = db.load(node_ids[4]);

        assert!(node0.is_ok());
        assert!(node1.is_err()); // Deleted
        assert!(node2.is_ok());
        assert!(node3.is_err()); // Deleted
        assert!(node4.is_ok());
    }

    Ok(())
}

/// Test scenario: Connection persistence
///
/// This test verifies that connections between nodes are properly
/// persisted to storage.
#[test]
fn test_storage_connection_persistence() -> Result<()> {
    // Create a temporary directory for our test
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test_connections.db");

    // Create nodes and connections
    let (node1_id, node2_id, node3_id);
    {
        let db = Database::file_based(&db_path)?;

        // Create 3 nodes
        let mut node1 = MemoryNode::new(vec![1.0, 0.0, 0.0]);
        node1.set_attribute(
            "name".to_string(),
            AttributeValue::String("Node 1".to_string()),
        );
        node1_id = db.save(&node1)?;

        let mut node2 = MemoryNode::new(vec![0.0, 1.0, 0.0]);
        node2.set_attribute(
            "name".to_string(),
            AttributeValue::String("Node 2".to_string()),
        );
        node2_id = db.save(&node2)?;

        let mut node3 = MemoryNode::new(vec![0.0, 0.0, 1.0]);
        node3.set_attribute(
            "name".to_string(),
            AttributeValue::String("Node 3".to_string()),
        );
        node3_id = db.save(&node3)?;

        // Create connections
        db.connect(node1_id, node2_id, "Related".to_string(), 0.9)?;
        db.connect(node2_id, node3_id, "Next".to_string(), 0.8)?;
        db.connect(node3_id, node1_id, "Previous".to_string(), 0.7)?;

        // Verify connections exist
        let node1_connections = db.get_connections(node1_id, None)?;
        assert_eq!(node1_connections.len(), 1);
        assert_eq!(node1_connections[0].target_id, node2_id);
    }

    // Verify connections persisted
    {
        let db = Database::file_based(&db_path)?;

        // Check node1's connections
        let node1_connections = db.get_connections(node1_id, None)?;
        assert_eq!(node1_connections.len(), 1);
        assert_eq!(node1_connections[0].target_id, node2_id);
        assert_eq!(node1_connections[0].relationship_type, "Related");
        assert_eq!(node1_connections[0].strength, 0.9);

        // Check node2's connections
        let node2_connections = db.get_connections(node2_id, None)?;
        assert_eq!(node2_connections.len(), 1);
        assert_eq!(node2_connections[0].target_id, node3_id);
        assert_eq!(node2_connections[0].relationship_type, "Next");
        assert_eq!(node2_connections[0].strength, 0.8);

        // Check node3's connections
        let node3_connections = db.get_connections(node3_id, None)?;
        assert_eq!(node3_connections.len(), 1);
        assert_eq!(node3_connections[0].target_id, node1_id);
        assert_eq!(node3_connections[0].relationship_type, "Previous");
        assert_eq!(node3_connections[0].strength, 0.7);
    }

    // Test connection modification
    {
        let db = Database::file_based(&db_path)?;

        // Disconnect node1 -> node2
        db.disconnect(node1_id, node2_id)?;

        // Connect node1 -> node3 instead
        db.connect(node1_id, node3_id, "Direct".to_string(), 0.95)?;

        // Verify changes
        let node1_connections = db.get_connections(node1_id, None)?;
        assert_eq!(node1_connections.len(), 1);
        assert_eq!(node1_connections[0].target_id, node3_id);
        assert_eq!(node1_connections[0].relationship_type, "Direct");
        assert_eq!(node1_connections[0].strength, 0.95);
    }

    // Verify connection changes persisted
    {
        let db = Database::file_based(&db_path)?;

        // Check node1's connections again
        let node1_connections = db.get_connections(node1_id, None)?;
        assert_eq!(node1_connections.len(), 1);
        assert_eq!(node1_connections[0].target_id, node3_id);
        assert_eq!(node1_connections[0].relationship_type, "Direct");
        assert_eq!(node1_connections[0].strength, 0.95);
    }

    Ok(())
}

/// Test scenario: Single-file vs multi-file storage
///
/// This test verifies that both the single-file and multi-file
/// storage backends work correctly and consistently.
#[test]
fn test_storage_single_file_vs_multi_file() -> Result<()> {
    // Create temporary directories for our test
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let single_file_path = temp_dir.path().join("single_file.db");
    let multi_file_path = temp_dir.path().join("multi_file");

    // Generate test data - same data for both storages
    let num_nodes = 10;
    let node_ids = Vec::with_capacity(num_nodes);

    // Setup: Create identical data in both databases
    {
        let single_db = Database::single_file_based(&single_file_path)?;
        let multi_db = Database::file_based(&multi_file_path)?;

        // Add identical nodes to both
        for i in 0..num_nodes {
            let mut node = MemoryNode::new(vec![i as f32, (i * 2) as f32]);
            node.set_attribute("index".to_string(), AttributeValue::Integer(i as i64));

            // Use the same ID for both databases
            let id = Uuid::new_v4();
            node.set_id(id);

            single_db.save(&node)?;
            multi_db.save(&node)?;
        }

        // Add some connections
        if let Some(ids) = single_db.list_all()?.get(0..3) {
            let id1 = ids[0];
            let id2 = ids[1];
            let id3 = ids[2];

            // Same connections in both DBs
            single_db.connect(id1, id2, "Link".to_string(), 0.9)?;
            single_db.connect(id2, id3, "Link".to_string(), 0.8)?;

            multi_db.connect(id1, id2, "Link".to_string(), 0.9)?;
            multi_db.connect(id2, id3, "Link".to_string(), 0.8)?;
        }
    }

    // Verify: Both databases have the same content
    {
        let single_db = Database::single_file_based(&single_file_path)?;
        let multi_db = Database::file_based(&multi_file_path)?;

        // Check node count
        assert_eq!(single_db.len()?, multi_db.len()?);

        // Get all IDs from both
        let single_ids: HashSet<_> = single_db.list_all()?.into_iter().collect();
        let multi_ids: HashSet<_> = multi_db.list_all()?.into_iter().collect();

        // Verify same IDs in both
        assert_eq!(single_ids, multi_ids);

        // Check each node
        for id in &single_ids {
            let single_node = single_db.load(*id)?;
            let multi_node = multi_db.load(*id)?;

            // Verify identical attributes
            assert_eq!(
                single_node.get_attribute("index").unwrap().to_string(),
                multi_node.get_attribute("index").unwrap().to_string()
            );

            // Verify identical embeddings
            assert_eq!(single_node.embedding(), multi_node.embedding());

            // Verify identical connections
            let single_connections = single_db.get_connections(*id, None)?;
            let multi_connections = multi_db.get_connections(*id, None)?;

            assert_eq!(single_connections.len(), multi_connections.len());

            if !single_connections.is_empty() {
                assert_eq!(
                    single_connections[0].target_id,
                    multi_connections[0].target_id
                );
                assert_eq!(
                    single_connections[0].relationship_type,
                    multi_connections[0].relationship_type
                );
                assert_eq!(
                    single_connections[0].strength,
                    multi_connections[0].strength
                );
            }
        }
    }

    // Test consistency after updates
    {
        let single_db = Database::single_file_based(&single_file_path)?;
        let multi_db = Database::file_based(&multi_file_path)?;

        // Get a node to update
        let test_id = single_db.list_all()?[0];

        // Update in both
        let mut single_node = single_db.load(test_id)?;
        let mut multi_node = multi_db.load(test_id)?;

        single_node.set_attribute("updated".to_string(), AttributeValue::Boolean(true));
        multi_node.set_attribute("updated".to_string(), AttributeValue::Boolean(true));

        single_db.save(&single_node)?;
        multi_db.save(&multi_node)?;
    }

    // Verify updates
    {
        let single_db = Database::single_file_based(&single_file_path)?;
        let multi_db = Database::file_based(&multi_file_path)?;

        // Get the updated node
        let test_id = single_db.list_all()?[0];

        let single_node = single_db.load(test_id)?;
        let multi_node = multi_db.load(test_id)?;

        // Verify both have the updated attribute
        assert_eq!(
            single_node.get_attribute("updated").unwrap().to_string(),
            "true"
        );
        assert_eq!(
            multi_node.get_attribute("updated").unwrap().to_string(),
            "true"
        );
    }

    Ok(())
}

/// Test scenario: Storage recovery after corruption
///
/// This test verifies that the storage engine can handle and
/// recover from some types of corruption.
#[test]
fn test_storage_recovery_from_corruption() -> Result<()> {
    // Create a temporary directory for our test
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test_corruption");

    // Create initial database
    let node_ids;
    {
        let db = Database::file_based(&db_path)?;

        // Add 10 nodes
        let mut ids = Vec::new();
        for i in 0..10 {
            let mut node = MemoryNode::new(vec![i as f32]);
            node.set_attribute("value".to_string(), AttributeValue::Integer(i as i64));
            ids.push(db.save(&node)?);
        }
        node_ids = ids;
    }

    // Now, corrupt the database by deleting some node files
    // This simulates a partial corruption where some data is lost
    if let Some(id) = node_ids.get(3) {
        let node_path = db_path.join(format!("{}/{}.json", &id.to_string()[0..2], id));
        if node_path.exists() {
            std::fs::remove_file(node_path).expect("Failed to delete file for corruption test");
        }
    }

    // Try to open the database again
    {
        let db = Database::file_based(&db_path)?;

        // We should still be able to access the database
        let count = db.len()?;
        assert_eq!(count, 9); // One node less than the original 10

        // Verify we can access the remaining nodes
        for (i, id) in node_ids.iter().enumerate() {
            if i != 3 {
                // Skip the corrupted node
                let result = db.load(*id);
                assert!(
                    result.is_ok(),
                    "Failed to load node {}: {:?}",
                    i,
                    result.err()
                );
            }
        }

        // The corrupted node should return an error
        let corrupted_result = db.load(node_ids[3]);
        assert!(corrupted_result.is_err());
    }

    Ok(())
}
