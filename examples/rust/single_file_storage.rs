use engramdb::{Database, MemoryNode};

fn main() -> engramdb::Result<()> {
    // Create a database with single-file storage
    println!("Creating database with single-file storage...");
    let mut db = Database::single_file("./test_single_file.engramdb")?;
    
    // Create some test memory nodes
    println!("Creating memory nodes...");
    let node1 = MemoryNode::new(vec![0.1, 0.2, 0.3]);
    let node2 = MemoryNode::new(vec![0.4, 0.5, 0.6]);
    let node3 = MemoryNode::new(vec![0.7, 0.8, 0.9]);
    
    // Save the nodes to the database
    let id1 = db.save(&node1)?;
    let id2 = db.save(&node2)?;
    let id3 = db.save(&node3)?;
    
    println!("Saved three memory nodes:");
    println!("  Node 1: {}", id1);
    println!("  Node 2: {}", id2);
    println!("  Node 3: {}", id3);
    
    // Create some connections between nodes
    println!("\nCreating connections...");
    db.connect(id1, id2, "Association".to_string(), 0.8)?;
    db.connect(id2, id3, "Causation".to_string(), 0.9)?;
    
    // Add some attributes
    println!("Adding attributes...");
    let mut node1 = db.load(id1)?;
    node1.set_attribute("name".to_string(), 
        engramdb::core::AttributeValue::String("First node".to_string()));
    db.save(&node1)?;
    
    // List all memory nodes
    println!("\nListing all memory nodes:");
    for id in db.list_all()? {
        let node = db.load(id)?;
        print!("  Node {}: ", id);
        
        // Check if it has a name attribute
        if let Some(engramdb::core::AttributeValue::String(name)) = node.get_attribute("name") {
            println!("{}", name);
        } else {
            println!("unnamed");
        }
        
        // List its connections
        let connections = db.get_connections(id, None)?;
        if !connections.is_empty() {
            println!("    Connections:");
            for conn in connections {
                println!("      -> {} ({}, strength: {})", 
                    conn.target_id, conn.type_name, conn.strength);
            }
        }
    }
    
    // Check vector search
    println!("\nPerforming vector search...");
    let results = db.search_similar(&[0.1, 0.2, 0.3], 2, 0.0, None, None)?;
    println!("  Results for similarity search:");
    for (id, similarity) in results {
        println!("    Node {}: similarity {:.2}", id, similarity);
    }
    
    // Close and reopen the database to demonstrate persistence
    drop(db); // Explicit drop to ensure everything is flushed
    println!("\nDatabase closed. Reopening...");
    
    let db2 = Database::single_file("./test_single_file.engramdb")?;
    println!("Database reopened, contains {} nodes", db2.len()?);
    
    // Use the storage type enum for configuration
    println!("\nCreating database with configuration...");
    let config = engramdb::DatabaseConfig {
        storage_type: engramdb::StorageType::SingleFile,
        storage_path: Some("./test_config.engramdb".to_string()),
        cache_size: 50,
    };
    
    let _db3 = Database::new(config)?;
    println!("Database created successfully with custom configuration");
    
    Ok(())
}