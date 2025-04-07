use rtamp::core::{AttributeValue, MemoryNode};
use rtamp::storage::{FileStorageEngine, MemoryStorageEngine, StorageEngine};
use rtamp::vector::VectorIndex;
use std::error::Error;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize in-memory storage
    let mut memory_storage = MemoryStorageEngine::new();
    println!("In-memory storage initialized");
    
    // Initialize file storage
    let storage_dir = PathBuf::from("./tmp_storage_example");
    let mut file_storage = FileStorageEngine::new(&storage_dir)?;
    println!("File storage initialized at: {}", storage_dir.display());
    
    // Initialize a shared vector index
    let mut vector_index = VectorIndex::new();
    
    // Create sample memories
    println!("\nCreating sample memories...");
    
    // Memory 1: Simple note
    let mut node1 = MemoryNode::new(vec![0.1, 0.2, 0.3, 0.4]);
    node1.set_attribute("title".to_string(), AttributeValue::String("Important note".to_string()));
    node1.set_attribute("content".to_string(), AttributeValue::String("Remember to update the documentation".to_string()));
    
    // Memory 2: Task
    let mut node2 = MemoryNode::new(vec![0.5, 0.4, 0.3, 0.2]);
    node2.set_attribute("title".to_string(), AttributeValue::String("Complete project".to_string()));
    node2.set_attribute("due_date".to_string(), AttributeValue::String("2023-12-31".to_string()));
    node2.set_attribute("priority".to_string(), AttributeValue::Float(0.8));
    
    // Save to in-memory storage
    println!("\nSaving to in-memory storage...");
    memory_storage.save(&node1)?;
    memory_storage.save(&node2)?;
    
    // Add to vector index
    vector_index.add(&node1)?;
    vector_index.add(&node2)?;
    
    // List all memories in memory storage
    let memory_ids = memory_storage.list_all()?;
    println!("In-memory storage contains {} memories", memory_ids.len());
    
    // Read data from memory storage
    for id in &memory_ids {
        let node = memory_storage.load(*id)?;
        print_memory(&node);
    }
    
    // Save from memory to file
    println!("\nPersisting memories to file storage...");
    for id in &memory_ids {
        let node = memory_storage.load(*id)?;
        file_storage.save(&node)?;
    }
    
    // Verify memories in file storage
    let file_ids = file_storage.list_all()?;
    println!("File storage contains {} memories", file_ids.len());
    
    // Clear memory storage to demonstrate loading from file
    println!("\nClearing in-memory storage...");
    memory_storage.clear();
    assert_eq!(memory_storage.list_all()?.len(), 0);
    println!("In-memory storage now contains {} memories", memory_storage.list_all()?.len());
    
    // Load memories back from file to memory
    println!("\nLoading memories from file storage back to memory...");
    for id in &file_ids {
        let node = file_storage.load(*id)?;
        memory_storage.save(&node)?;
    }
    
    println!("In-memory storage now contains {} memories", memory_storage.list_all()?.len());
    
    // Demonstrate searching with vector index
    println!("\nSearching for memories with vector similarity...");
    let query_vector = vec![0.2, 0.3, 0.3, 0.4]; // More similar to node1
    let results = vector_index.search(&query_vector, 2, 0.0)?;
    
    println!("Found {} similar memories:", results.len());
    for (id, similarity) in &results {
        // Load from memory storage (could also load from file_storage)
        let node = memory_storage.load(*id)?;
        println!("  Memory: {} (similarity: {:.4})",
            get_title(&node),
            similarity
        );
    }
    
    // Clean up
    println!("\nCleaning up...");
    for id in file_ids {
        file_storage.delete(id)?;
    }
    
    std::fs::remove_dir_all(storage_dir)?;
    println!("Done!");
    
    Ok(())
}

fn print_memory(node: &MemoryNode) {
    println!("  Memory: {}", get_title(node));
    
    for (key, value) in node.attributes() {
        if key != "title" {
            println!("    {}: {}", key, format_attribute_value(value));
        }
    }
}

fn get_title(node: &MemoryNode) -> String {
    node.get_attribute("title")
        .and_then(|v| if let AttributeValue::String(s) = v { Some(s.as_str()) } else { None })
        .unwrap_or("Untitled")
        .to_string()
}

fn format_attribute_value(value: &AttributeValue) -> String {
    match value {
        AttributeValue::String(s) => s.clone(),
        AttributeValue::Integer(i) => i.to_string(),
        AttributeValue::Float(f) => format!("{:.2}", f),
        AttributeValue::Boolean(b) => b.to_string(),
        AttributeValue::List(list) => {
            let items: Vec<String> = list.iter()
                .map(format_attribute_value)
                .collect();
            format!("[{}]", items.join(", "))
        },
        AttributeValue::Map(map) => {
            let items: Vec<String> = map.iter()
                .map(|(k, v)| format!("{}: {}", k, format_attribute_value(v)))
                .collect();
            format!("{{{}}}", items.join(", "))
        },
    }
}