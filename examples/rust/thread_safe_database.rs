use engramdb::{
    core::MemoryNode,
    vector::ThreadSafeDatabase,
    core::AttributeValue,
    Result,
};
use std::sync::Arc;
use std::thread;
use uuid::Uuid;

/// Creates memory nodes in the database
fn create_memory_nodes(db: &ThreadSafeDatabase, count: usize) -> Result<Vec<Uuid>> {
    let mut ids = Vec::with_capacity(count);
    
    for i in 0..count {
        // Create a memory node with a simple embedding
        let embedding = vec![i as f32, (i + 1) as f32, (i + 2) as f32];
        let mut node = MemoryNode::new(embedding);
        
        // Add some attributes
        node.set_attribute(
            "name".to_string(), 
            AttributeValue::String(format!("Node {}", i))
        );
        node.set_attribute(
            "value".to_string(), 
            AttributeValue::Integer(i as i64)
        );
        
        // Save to database
        let node_id = db.save(&node)?;
        ids.push(node_id);
    }
    
    Ok(ids)
}

/// Worker function for thread operations
fn worker_function(
    db: Arc<ThreadSafeDatabase>, 
    thread_index: usize, 
    iterations: usize
) -> Result<()> {
    println!("Thread {} starting", thread_index);
    
    // Create some nodes specific to this thread
    let mut thread_nodes = Vec::with_capacity(iterations);
    
    for i in 0..iterations {
        // Create a memory node with a unique embedding
        let embedding = vec![(thread_index * 100 + i) as f32, 0.0, 0.0];
        let mut node = MemoryNode::new(embedding);
        
        // Add thread-specific metadata
        node.set_attribute(
            "thread".to_string(), 
            AttributeValue::Integer(thread_index as i64)
        );
        node.set_attribute(
            "iteration".to_string(), 
            AttributeValue::Integer(i as i64)
        );
        
        // Save to database
        let node_id = db.save(&node)?;
        thread_nodes.push(node_id);
    }
    
    // Now load and verify each node
    for (i, node_id) in thread_nodes.iter().enumerate() {
        let loaded_node = db.load(*node_id)?;
        
        // Verify the metadata
        if let Some(AttributeValue::Integer(thread_val)) = loaded_node.get_attribute("thread") {
            assert_eq!(*thread_val, thread_index as i64);
        } else {
            panic!("Missing or invalid thread attribute");
        }
        
        if let Some(AttributeValue::Integer(iter_val)) = loaded_node.get_attribute("iteration") {
            assert_eq!(*iter_val, i as i64);
        } else {
            panic!("Missing or invalid iteration attribute");
        }
    }
    
    // Create some connections between nodes
    if thread_nodes.len() >= 2 {
        for i in 0..thread_nodes.len() - 1 {
            db.connect(
                thread_nodes[i], 
                thread_nodes[i + 1], 
                "Sequence".to_string(), 
                0.9
            )?;
        }
    }
    
    println!("Thread {} completed", thread_index);
    Ok(())
}

fn main() -> Result<()> {
    // Create a thread-safe in-memory database with HNSW
    let db = Arc::new(ThreadSafeDatabase::in_memory_with_hnsw());
    
    // Create some initial data
    let initial_ids = create_memory_nodes(&db, 10)?;
    println!("Created {} initial nodes", initial_ids.len());
    
    // Create multiple threads to operate on the database
    let mut handles = Vec::new();
    
    for i in 0..5 {
        let db_clone = db.clone();
        let handle = thread::spawn(move || {
            worker_function(db_clone, i, 20).unwrap();
        });
        handles.push(handle);
    }
    
    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Verify the final state of the database
    let count = db.len()?;
    println!("Database contains {} memory nodes", count);
    
    // Get all node IDs
    let all_ids = db.list_all()?;
    println!("Total nodes: {}", all_ids.len());
    
    // Try a vector search using one of the thread's specific vectors
    let query_vector = vec![200.0, 0.0, 0.0]; // From thread 2, iteration 0
    let results = db.search_similar(&query_vector, 5, 0.1, None, None)?;
    
    println!("\nSearch results:");
    for (node_id, similarity) in results {
        let node = db.load(node_id)?;
        
        let thread = if let Some(AttributeValue::Integer(val)) = node.get_attribute("thread") {
            val.to_string()
        } else {
            "N/A".to_string()
        };
        
        let iter_val = if let Some(AttributeValue::Integer(val)) = node.get_attribute("iteration") {
            val.to_string()
        } else {
            "N/A".to_string()
        };
        
        println!(
            "  Node {}: similarity={:.4}, thread={}, iteration={}", 
            node_id, similarity, thread, iter_val
        );
    }
    
    Ok(())
}