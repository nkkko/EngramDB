use rtamp::core::{AttributeValue, MemoryNode};
use rtamp::query::{AttributeFilter, QueryBuilder, TemporalFilter};
use rtamp::storage::FileStorageEngine;
use rtamp::vector::VectorIndex;
use rtamp::StorageEngine;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize storage in a temporary directory
    let storage_dir = PathBuf::from("./tmp_memories");
    let mut storage = FileStorageEngine::new(&storage_dir)?;
    println!("Storage initialized at: {}", storage_dir.display());
    
    // Initialize vector index
    let mut vector_index = VectorIndex::new();
    
    // Create example memories
    create_sample_memories(&mut storage, &mut vector_index)?;
    
    // List all memories
    let memory_ids = storage.list_all()?;
    println!("Created {} memories", memory_ids.len());
    
    // Query for memories by vector similarity
    println!("\nSearching for memories similar to 'meeting notes'...");
    let query_vector = vec![0.1, 0.3, 0.5, 0.2];
    let search_results = vector_index.search(&query_vector, 3, 0.5)?;
    
    println!("Found {} similar memories:", search_results.len());
    for (id, similarity) in &search_results {
        let node = storage.load(*id)?;
        print_memory(&node, *similarity);
    }
    
    // Query combining filters
    println!("\nSearching for important memories...");
    let importance_filter = AttributeFilter::greater_than(
        "importance".to_string(),
        AttributeValue::Float(0.7),
    );
    
    let time_filter = TemporalFilter::within_last(3600 * 24); // Last 24 hours
    
    let query = QueryBuilder::new()
        .with_vector(query_vector)
        .with_attribute_filter(importance_filter)
        .with_temporal_filter(time_filter);
    
    let query_results = query.execute(&vector_index, |id| storage.load(id))?;
    
    println!("Found {} important memories:", query_results.len());
    for node in &query_results {
        let similarity = search_results.iter()
            .find(|(id, _)| *id == node.id())
            .map(|(_, sim)| *sim)
            .unwrap_or(0.0);
            
        print_memory(node, similarity);
    }
    
    // Clean up
    println!("\nCleaning up...");
    for id in memory_ids {
        storage.delete(id)?;
    }
    
    std::fs::remove_dir_all(storage_dir)?;
    println!("Done!");
    
    Ok(())
}

fn create_sample_memories(
    storage: &mut impl StorageEngine,
    vector_index: &mut VectorIndex,
) -> Result<(), Box<dyn Error>> {
    // Memory 1: Meeting notes
    let mut node1 = MemoryNode::new(vec![0.1, 0.3, 0.5, 0.2]);
    node1.set_attribute("title".to_string(), AttributeValue::String("Meeting Notes".to_string()));
    node1.set_attribute("importance".to_string(), AttributeValue::Float(0.8));
    node1.set_attribute("category".to_string(), AttributeValue::String("work".to_string()));
    storage.save(&node1)?;
    vector_index.add(&node1)?;
    
    // Memory 2: Shopping list
    let mut node2 = MemoryNode::new(vec![0.8, 0.1, 0.0, 0.3]);
    node2.set_attribute("title".to_string(), AttributeValue::String("Shopping List".to_string()));
    node2.set_attribute("importance".to_string(), AttributeValue::Float(0.3));
    node2.set_attribute("category".to_string(), AttributeValue::String("personal".to_string()));
    storage.save(&node2)?;
    vector_index.add(&node2)?;
    
    // Memory 3: Project idea
    let mut node3 = MemoryNode::new(vec![0.2, 0.4, 0.4, 0.1]);
    node3.set_attribute("title".to_string(), AttributeValue::String("Project Idea".to_string()));
    node3.set_attribute("importance".to_string(), AttributeValue::Float(0.9));
    node3.set_attribute("category".to_string(), AttributeValue::String("work".to_string()));
    storage.save(&node3)?;
    vector_index.add(&node3)?;
    
    Ok(())
}

fn print_memory(node: &MemoryNode, similarity: f32) {
    println!("  Memory: {} (similarity: {:.4})", 
        node.get_attribute("title")
            .and_then(|v| if let AttributeValue::String(s) = v { Some(s) } else { None })
            .unwrap_or(&"Untitled".to_string()),
        similarity
    );
    
    println!("    Importance: {}", 
        node.get_attribute("importance")
            .and_then(|v| if let AttributeValue::Float(f) = v { Some(f) } else { None })
            .unwrap_or(&0.0)
    );
    
    println!("    Category: {}", 
        node.get_attribute("category")
            .and_then(|v| if let AttributeValue::String(s) = v { Some(s) } else { None })
            .unwrap_or(&"Uncategorized".to_string())
    );
}