use engramdb::{
    core::{AttributeValue, MemoryNode},
    vector::ThreadSafeDatabase,
    Result,
};
use std::sync::{Arc, Barrier};
use std::thread;
use std::time::Duration;
use uuid::Uuid;

/// Test scenario: Load test with concurrent operations
///
/// This test creates a shared database with multiple threads performing different operations:
/// - Reader threads continually reading data
/// - Writer threads adding new data
/// - Modifier threads updating existing data
/// - Search threads performing vector searches
/// - Connection threads managing relationships between nodes
#[test]
fn test_thread_safety_concurrent_operation_load() -> Result<()> {
    // Parameters
    let num_reader_threads = 4;
    let num_writer_threads = 2;
    let num_modifier_threads = 2;
    let num_search_threads = 2;
    let num_connection_threads = 2;
    let total_threads = num_reader_threads
        + num_writer_threads
        + num_modifier_threads
        + num_search_threads
        + num_connection_threads;
    let operations_per_thread = 50;
    let initial_nodes = 20;

    // Create database
    let db = Arc::new(ThreadSafeDatabase::in_memory_with_hnsw());

    // Set up initial data
    println!("Setting up initial test data...");
    let mut initial_ids = Vec::with_capacity(initial_nodes);
    for i in 0..initial_nodes {
        let embedding = vec![i as f32, (i * 2) as f32, (i * 3) as f32];
        let mut node = MemoryNode::new(embedding);
        node.set_attribute(
            "name".to_string(),
            AttributeValue::String(format!("Node {}", i)),
        );
        node.set_attribute("value".to_string(), AttributeValue::Integer(i as i64));
        let id = db.save(&node)?;
        initial_ids.push(id);
    }

    // Create some initial connections
    for i in 0..initial_ids.len() - 1 {
        db.connect(initial_ids[i], initial_ids[i + 1], "Next".to_string(), 0.9)?;
    }

    // Barrier to synchronize thread start
    let barrier = Arc::new(Barrier::new(total_threads));

    // Create tracking for thread handles
    let mut handles = Vec::with_capacity(total_threads);

    // 1. Reader threads - read existing nodes
    for thread_id in 0..num_reader_threads {
        let db_clone = db.clone();
        let ids_clone = initial_ids.clone();
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || -> Result<()> {
            println!("Reader thread {} waiting to start", thread_id);
            barrier_clone.wait();
            println!("Reader thread {} starting", thread_id);

            for _ in 0..operations_per_thread {
                // Pick a random ID to read
                let idx = (thread_id + thread::current().id().as_u64() as usize) % ids_clone.len();
                let id = ids_clone[idx];

                // Read the node
                let node = db_clone.load(id)?;
                assert_eq!(node.id(), id);

                // Small random delay to simulate work
                if thread_id % 2 == 0 {
                    thread::sleep(Duration::from_millis(1));
                }
            }

            println!("Reader thread {} completed", thread_id);
            Ok(())
        });

        handles.push(handle);
    }

    // 2. Writer threads - add new nodes
    for thread_id in 0..num_writer_threads {
        let db_clone = db.clone();
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || -> Result<()> {
            println!("Writer thread {} waiting to start", thread_id);
            barrier_clone.wait();
            println!("Writer thread {} starting", thread_id);

            let mut created_ids = Vec::with_capacity(operations_per_thread);

            for i in 0..operations_per_thread {
                // Create a unique embedding
                let base = thread_id * 1000 + i;
                let embedding = vec![base as f32, (base * 2) as f32, (base * 3) as f32];

                // Create node
                let mut node = MemoryNode::new(embedding);
                node.set_attribute(
                    "name".to_string(),
                    AttributeValue::String(format!("Thread {} Node {}", thread_id, i)),
                );
                node.set_attribute(
                    "thread".to_string(),
                    AttributeValue::Integer(thread_id as i64),
                );
                node.set_attribute("index".to_string(), AttributeValue::Integer(i as i64));

                // Save node
                let id = db_clone.save(&node)?;
                created_ids.push(id);

                // Small random delay to simulate work
                if i % 5 == 0 {
                    thread::sleep(Duration::from_millis(2));
                }
            }

            println!(
                "Writer thread {} completed, created {} nodes",
                thread_id,
                created_ids.len()
            );
            Ok(())
        });

        handles.push(handle);
    }

    // 3. Modifier threads - update existing nodes
    for thread_id in 0..num_modifier_threads {
        let db_clone = db.clone();
        let ids_clone = initial_ids.clone();
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || -> Result<()> {
            println!("Modifier thread {} waiting to start", thread_id);
            barrier_clone.wait();
            println!("Modifier thread {} starting", thread_id);

            for i in 0..operations_per_thread {
                // Pick a node to modify
                let idx = (thread_id + i) % ids_clone.len();
                let id = ids_clone[idx];

                // Load the node
                let mut node = db_clone.load(id)?;

                // Modify an attribute
                node.set_attribute(
                    "modified_by".to_string(),
                    AttributeValue::String(format!("Thread {}", thread_id)),
                );
                node.set_attribute(
                    "modification_count".to_string(),
                    AttributeValue::Integer(i as i64),
                );

                // Save the modified node
                db_clone.save(&node)?;

                // Small random delay
                if i % 3 == 0 {
                    thread::sleep(Duration::from_millis(3));
                }
            }

            println!("Modifier thread {} completed", thread_id);
            Ok(())
        });

        handles.push(handle);
    }

    // 4. Search threads - perform vector searches
    for thread_id in 0..num_search_threads {
        let db_clone = db.clone();
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || -> Result<()> {
            println!("Search thread {} waiting to start", thread_id);
            barrier_clone.wait();
            println!("Search thread {} starting", thread_id);

            for i in 0..operations_per_thread {
                // Create a query vector
                let base = thread_id * 10 + (i % 10);
                let query = vec![base as f32, (base * 2) as f32, (base * 3) as f32];

                // Perform search
                let results = db_clone.search_similar(&query, 5, 0.0, None, None)?;

                // Verify we got reasonable results
                assert!(results.len() <= 5);

                // Small random delay
                if i % 7 == 0 {
                    thread::sleep(Duration::from_millis(2));
                }
            }

            println!("Search thread {} completed", thread_id);
            Ok(())
        });

        handles.push(handle);
    }

    // 5. Connection threads - manage connections
    for thread_id in 0..num_connection_threads {
        let db_clone = db.clone();
        let ids_clone = initial_ids.clone();
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || -> Result<()> {
            println!("Connection thread {} waiting to start", thread_id);
            barrier_clone.wait();
            println!("Connection thread {} starting", thread_id);

            for i in 0..operations_per_thread {
                // Choose random source and target
                let source_idx = (thread_id + i) % ids_clone.len();
                let target_idx = (source_idx + 1) % ids_clone.len();

                let source_id = ids_clone[source_idx];
                let target_id = ids_clone[target_idx];

                // Create connection
                if i % 2 == 0 {
                    // Add connection
                    db_clone.connect(
                        source_id,
                        target_id,
                        format!("Connection{}", i),
                        0.8 + (i as f32 * 0.01),
                    )?;
                } else {
                    // Get connections to verify
                    let connections = db_clone.get_connections(source_id, None)?;

                    // Check if any connections exist to target
                    let has_connection = connections.iter().any(|conn| conn.target_id == target_id);

                    if has_connection {
                        // Remove one connection
                        let _ = db_clone.disconnect(source_id, target_id);
                    }
                }

                // Small random delay
                if i % 4 == 0 {
                    thread::sleep(Duration::from_millis(2));
                }
            }

            println!("Connection thread {} completed", thread_id);
            Ok(())
        });

        handles.push(handle);
    }

    // Wait for all threads to complete and collect results
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(result) => {
                if let Err(e) = result {
                    println!("Thread {} failed with error: {}", i, e);
                    return Err(e);
                }
            }
            Err(e) => {
                println!("Thread {} panicked: {:?}", i, e);
                return Err(engramdb::error::EngramDbError::Other(format!(
                    "Thread panic: {:?}",
                    e
                )));
            }
        }
    }

    // Verify final database state
    let final_count = db.len()?;
    println!("Final database contains {} nodes", final_count);
    assert!(final_count >= initial_nodes as usize + (num_writer_threads * operations_per_thread));

    Ok(())
}

/// Test scenario: Deadlock detection
///
/// This test tries to create conditions that might lead to deadlocks and
/// verifies that the database properly handles them.
#[test]
fn test_thread_safety_deadlock_prevention() -> Result<()> {
    let db = Arc::new(ThreadSafeDatabase::in_memory());

    // Create some initial data
    let mut node1 = MemoryNode::new(vec![1.0, 0.0, 0.0]);
    node1.set_attribute(
        "name".to_string(),
        AttributeValue::String("Node 1".to_string()),
    );
    let id1 = db.save(&node1)?;

    let mut node2 = MemoryNode::new(vec![0.0, 1.0, 0.0]);
    node2.set_attribute(
        "name".to_string(),
        AttributeValue::String("Node 2".to_string()),
    );
    let id2 = db.save(&node2)?;

    // Connect the nodes
    db.connect(id1, id2, "Related".to_string(), 0.9)?;

    // Create two threads that will try to update the nodes and their connections
    // in opposite orders, which might lead to deadlocks in a naive implementation
    let barrier = Arc::new(Barrier::new(2));

    // Thread 1: Update node1, then update connection from node1 to node2
    let db_clone1 = db.clone();
    let barrier_clone1 = barrier.clone();
    let id1_clone = id1;
    let id2_clone = id2;

    let handle1 = thread::spawn(move || -> Result<()> {
        barrier_clone1.wait();

        // First, update node1
        let mut node = db_clone1.load(id1_clone)?;
        node.set_attribute(
            "updated_by".to_string(),
            AttributeValue::String("Thread 1".to_string()),
        );
        db_clone1.save(&node)?;

        // Small delay to increase chance of deadlock
        thread::sleep(Duration::from_millis(10));

        // Then, update the connection
        db_clone1.disconnect(id1_clone, id2_clone)?;
        db_clone1.connect(id1_clone, id2_clone, "UpdatedByThread1".to_string(), 0.95)?;

        Ok(())
    });

    // Thread 2: Update node2, then update connection from node1 to node2
    let db_clone2 = db.clone();
    let barrier_clone2 = barrier.clone();
    let id1_clone = id1;
    let id2_clone = id2;

    let handle2 = thread::spawn(move || -> Result<()> {
        barrier_clone2.wait();

        // First, update node2
        let mut node = db_clone2.load(id2_clone)?;
        node.set_attribute(
            "updated_by".to_string(),
            AttributeValue::String("Thread 2".to_string()),
        );
        db_clone2.save(&node)?;

        // Small delay to increase chance of deadlock
        thread::sleep(Duration::from_millis(10));

        // Then, get the connection information
        let connections = db_clone2.get_connections(id1_clone, None)?;

        // Use the connection info somehow to ensure it's loaded
        assert!(connections.iter().any(|c| c.target_id == id2_clone));

        Ok(())
    });

    // Wait for both threads - if we get deadlocks, this will hang
    let result1 = handle1.join().expect("Thread 1 panicked")?;
    let result2 = handle2.join().expect("Thread 2 panicked")?;

    // If we got here, no deadlock occurred
    Ok(())
}

/// Test scenario: Multiple database instances sharing storage
///
/// This test creates multiple ThreadSafeDatabase instances that share the same
/// storage, and verifies that they can all see the same data.
#[test]
fn test_thread_safety_shared_storage() -> Result<()> {
    use std::path::Path;
    use tempfile::tempdir;

    // Create a temporary directory for our test database
    let temp_dir = tempdir().expect("Failed to create temp directory");
    let db_path = temp_dir.path().join("test_shared.db");

    // Create a database and add some data
    {
        let db = ThreadSafeDatabase::file_based_with_hnsw(&db_path)?;

        // Add some test data
        for i in 0..5 {
            let mut node = MemoryNode::new(vec![i as f32, i as f32 * 2.0, i as f32 * 3.0]);
            node.set_attribute("index".to_string(), AttributeValue::Integer(i));
            db.save(&node)?;
        }
    }

    // Create a pool and multiple connections to the same database
    let pool = ThreadSafeDatabase::create_connection_pool(&db_path)?;
    let num_threads = 4;
    let ops_per_thread = 10;

    // Create our threads
    let mut handles = Vec::with_capacity(num_threads);
    let barrier = Arc::new(Barrier::new(num_threads));

    for thread_id in 0..num_threads {
        let pool_clone = pool.clone();
        let barrier_clone = barrier.clone();

        let handle = thread::spawn(move || -> Result<()> {
            // Get a connection from the pool
            let db = pool_clone.get_connection()?;

            // Wait for all threads to get their connection
            barrier_clone.wait();

            // Perform operations
            for i in 0..ops_per_thread {
                if i % 2 == 0 {
                    // Add a new node
                    let mut node = MemoryNode::new(vec![
                        thread_id as f32 * 100.0 + i as f32,
                        thread_id as f32 * 200.0 + i as f32,
                        thread_id as f32 * 300.0 + i as f32,
                    ]);
                    node.set_attribute(
                        "thread".to_string(),
                        AttributeValue::Integer(thread_id as i64),
                    );
                    node.set_attribute("op".to_string(), AttributeValue::Integer(i as i64));
                    db.save(&node)?;
                } else {
                    // Read all nodes
                    let ids = db.list_all()?;
                    assert!(!ids.is_empty());

                    // Load a couple of nodes
                    for idx in 0..std::cmp::min(3, ids.len()) {
                        let node = db.load(ids[idx])?;
                        assert_eq!(node.id(), ids[idx]);
                    }
                }

                thread::sleep(Duration::from_millis(1));
            }

            Ok(())
        });

        handles.push(handle);
    }

    // Wait for all threads to complete
    for (i, handle) in handles.into_iter().enumerate() {
        match handle.join() {
            Ok(result) => {
                if let Err(e) = result {
                    println!("Thread {} failed with error: {}", i, e);
                    return Err(e);
                }
            }
            Err(e) => {
                println!("Thread {} panicked: {:?}", i, e);
                return Err(engramdb::error::EngramDbError::Other(format!(
                    "Thread panic: {:?}",
                    e
                )));
            }
        }
    }

    // Open the database one last time to verify the state
    let final_db = ThreadSafeDatabase::file_based(&db_path)?;
    let final_count = final_db.len()?;
    assert_eq!(final_count, 5 + (num_threads * ops_per_thread / 2) as usize);

    // Successful completion
    Ok(())
}
