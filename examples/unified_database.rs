use rtamp::{Database, DatabaseConfig, MemoryNode};
use rtamp::core::AttributeValue;
use rtamp::query::{AttributeFilter, TemporalFilter};
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    // Create an in-memory database for testing
    let mut db = Database::in_memory();
    println!("Created in-memory database");
    
    // Add some test memories
    println!("\nAdding test memories...");
    add_test_memories(&mut db)?;
    
    // Query for all memories
    let all_memories = db.list_all()?;
    println!("Database contains {} memories", all_memories.len());
    
    // Vector similarity search
    println!("\nSearching for memories similar to 'project plan'...");
    let query_vector = vec![0.1, 0.3, 0.5, 0.1];
    let similar_results = db.search_similar(&query_vector, 5, 0.0)?;
    
    println!("Found {} similar memories:", similar_results.len());
    for (id, similarity) in &similar_results {
        let node = db.load(*id)?;
        println!("  {} (similarity: {:.4})",
            node.get_attribute("title")
                .and_then(|v| if let AttributeValue::String(s) = v { Some(s) } else { None })
                .unwrap_or(&"Untitled".to_string()),
            similarity
        );
    }
    
    // Combined query with filters
    println!("\nSearching for important work memories...");
    
    // Create attribute filter for "category" = "work"
    let category_filter = AttributeFilter::equals(
        "category".to_string(),
        AttributeValue::String("work".to_string()),
    );
    
    // Create attribute filter for "importance" > 0.7
    let importance_filter = AttributeFilter::greater_than(
        "importance".to_string(),
        AttributeValue::Float(0.7),
    );
    
    // Create temporal filter for recent memories (last 24 hours)
    let time_filter = TemporalFilter::within_last(24 * 60 * 60);
    
    // Execute the combined query
    let results = db.query()
        .with_vector(query_vector)
        .with_attribute_filter(category_filter)
        .with_attribute_filter(importance_filter)
        .with_temporal_filter(time_filter)
        .execute()?;
    
    println!("Found {} important work memories:", results.len());
    for node in &results {
        println!("  {}", 
            node.get_attribute("title")
                .and_then(|v| if let AttributeValue::String(s) = v { Some(s) } else { None })
                .unwrap_or(&"Untitled".to_string())
        );
        
        println!("    Importance: {}", 
            node.get_attribute("importance")
                .and_then(|v| if let AttributeValue::Float(f) = v { Some(f) } else { None })
                .unwrap_or(&0.0)
        );
    }
    
    // Now create a persistent database
    println!("\nCreating a persistent database...");
    let config = DatabaseConfig {
        use_memory_storage: false,
        storage_dir: Some("./tmp_database".to_string()),
        cache_size: 100,
    };
    
    let mut persistent_db = Database::new(config)?;
    
    // Transfer memories from in-memory to persistent
    println!("Transferring memories to persistent storage...");
    for id in db.list_all()? {
        let node = db.load(id)?;
        persistent_db.save(&node)?;
    }
    
    // Verify memories were transferred
    println!("Persistent database contains {} memories", persistent_db.list_all()?.len());
    
    // Clean up
    println!("\nCleaning up...");
    std::fs::remove_dir_all("./tmp_database")?;
    println!("Done!");
    
    Ok(())
}

fn add_test_memories(db: &mut Database) -> Result<(), Box<dyn Error>> {
    // Memory 1: Project Plan
    let mut node1 = MemoryNode::new(vec![0.1, 0.3, 0.5, 0.1]);
    node1.set_attribute("title".to_string(), AttributeValue::String("Project Plan".to_string()));
    node1.set_attribute("category".to_string(), AttributeValue::String("work".to_string()));
    node1.set_attribute("importance".to_string(), AttributeValue::Float(0.9));
    db.save(&node1)?;
    
    // Memory 2: Shopping List
    let mut node2 = MemoryNode::new(vec![0.8, 0.1, 0.05, 0.05]);
    node2.set_attribute("title".to_string(), AttributeValue::String("Shopping List".to_string()));
    node2.set_attribute("category".to_string(), AttributeValue::String("personal".to_string()));
    node2.set_attribute("importance".to_string(), AttributeValue::Float(0.4));
    db.save(&node2)?;
    
    // Memory 3: Meeting Notes
    let mut node3 = MemoryNode::new(vec![0.2, 0.4, 0.3, 0.1]);
    node3.set_attribute("title".to_string(), AttributeValue::String("Meeting Notes".to_string()));
    node3.set_attribute("category".to_string(), AttributeValue::String("work".to_string()));
    node3.set_attribute("importance".to_string(), AttributeValue::Float(0.8));
    db.save(&node3)?;
    
    // Memory 4: Birthday Reminder
    let mut node4 = MemoryNode::new(vec![0.7, 0.2, 0.05, 0.05]);
    node4.set_attribute("title".to_string(), AttributeValue::String("Birthday Reminder".to_string()));
    node4.set_attribute("category".to_string(), AttributeValue::String("personal".to_string()));
    node4.set_attribute("importance".to_string(), AttributeValue::Float(0.6));
    db.save(&node4)?;
    
    // Memory 5: Project Timeline
    let mut node5 = MemoryNode::new(vec![0.15, 0.35, 0.4, 0.1]);
    node5.set_attribute("title".to_string(), AttributeValue::String("Project Timeline".to_string()));
    node5.set_attribute("category".to_string(), AttributeValue::String("work".to_string()));
    node5.set_attribute("importance".to_string(), AttributeValue::Float(0.85));
    db.save(&node5)?;
    
    Ok(())
}