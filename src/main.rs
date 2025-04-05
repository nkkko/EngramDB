use engramdb::core::MemoryNode;
use engramdb::query::QueryBuilder;
use engramdb::storage::{FileStorageEngine, MemoryStorageEngine};
use engramdb::vector::VectorIndex;
use engramdb::StorageEngine;
use std::env;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    env_logger::init();

    println!("EngramDB: Engram Database");
    println!("-------------------------");

    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    
    // Check for in-memory flag
    let use_memory_storage = args.iter().any(|arg| arg == "--memory" || arg == "-m");
    
    // Get storage directory from arguments or use default
    let storage_dir = args.iter()
        .skip(1)
        .find(|arg| !arg.starts_with("-"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./memory_storage"));

    // Initialize appropriate storage engine
    let storage: Box<dyn StorageEngine> = if use_memory_storage {
        println!("Using in-memory storage");
        Box::new(MemoryStorageEngine::new())
    } else {
        println!("Using file storage at: {}", storage_dir.display());
        Box::new(FileStorageEngine::new(&storage_dir)?)
    };
    
    // Create a mutable reference to the storage engine
    let mut storage = storage;
    println!("Storage engine initialized");

    // Initialize vector index
    let mut vector_index = VectorIndex::new();
    println!("Vector index initialized");

    // Load existing memories
    let memory_ids = storage.list_all()?;
    println!("Found {} existing memories", memory_ids.len());

    for id in &memory_ids {
        let node = storage.load(*id)?;
        vector_index.add(&node)?;
    }

    // Demo: Create a new memory
    println!("\nCreating a demo memory...");
    let embeddings = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let mut memory = MemoryNode::new(embeddings);

    memory.set_attribute(
        "description".to_string(),
        engramdb::core::AttributeValue::String("This is a demo memory".to_string()),
    );
    memory.set_attribute(
        "importance".to_string(),
        engramdb::core::AttributeValue::Float(0.8),
    );

    let memory_id = memory.id();
    println!("Created memory with ID: {}", memory_id);

    // Save the memory
    storage.save(&memory)?;
    vector_index.add(&memory)?;
    println!("Memory saved to storage and indexed");

    // Demo: Query for similar memories
    println!("\nDemonstrating vector similarity search...");
    let query_vector = vec![0.15, 0.25, 0.35, 0.45, 0.55];
    let search_results = vector_index.search(&query_vector, 5, 0.0)?;

    println!("Found {} similar memories:", search_results.len());
    for (id, similarity) in search_results {
        println!("  Memory ID: {} (similarity: {:.4})", id, similarity);
    }

    // Demo: Use the query builder
    println!("\nDemonstrating query builder...");
    let query = QueryBuilder::new()
        .with_vector(query_vector.clone())
        .with_limit(5);

    let query_results = query.execute(&vector_index, |id| storage.load(id))?;

    println!("Query returned {} results:", query_results.len());
    for node in query_results {
        println!("  Memory ID: {}", node.id());
        if let Some(engramdb::core::AttributeValue::String(desc)) = node.get_attribute("description") {
            println!("    Description: {}", desc);
        }
    }

    println!("\nEngramDB demo completed successfully!");

    Ok(())
}
