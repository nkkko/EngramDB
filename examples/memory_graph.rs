use rtamp::core::{AttributeValue, Connection, MemoryNode};
use rtamp::core::connection::RelationshipType;
use rtamp::storage::FileStorageEngine;
use rtamp::StorageEngine;
use std::collections::HashMap;
use std::error::Error;
use std::path::PathBuf;
use uuid::Uuid;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize storage
    let storage_dir = PathBuf::from("./tmp_graph_memories");
    let mut storage = FileStorageEngine::new(&storage_dir)?;
    println!("Storage initialized at: {}", storage_dir.display());
    
    // Create a graph of connected memories
    println!("\nCreating a graph of memories...");
    let memory_ids = create_memory_graph(&mut storage)?;
    
    // Traverse the graph
    println!("\nTraversing the memory graph from the event memory:");
    let event_id = memory_ids.get("event").unwrap();
    traverse_graph(&storage, *event_id, 0)?;
    
    // Clean up
    println!("\nCleaning up...");
    for id in memory_ids.values() {
        storage.delete(*id)?;
    }
    
    std::fs::remove_dir_all(storage_dir)?;
    println!("Done!");
    
    Ok(())
}

fn create_memory_graph(storage: &mut impl StorageEngine) -> Result<HashMap<String, Uuid>, Box<dyn Error>> {
    let mut memory_ids = HashMap::new();
    
    // Create the "event" memory (trip to Paris)
    let mut event = MemoryNode::new(vec![0.1, 0.2, 0.3, 0.4]);
    event.set_attribute("title".to_string(), AttributeValue::String("Trip to Paris".to_string()));
    event.set_attribute("type".to_string(), AttributeValue::String("event".to_string()));
    event.set_attribute("date".to_string(), AttributeValue::String("2023-05-15".to_string()));
    
    // Create "location" memory
    let mut location = MemoryNode::new(vec![0.3, 0.3, 0.3, 0.3]);
    location.set_attribute("title".to_string(), AttributeValue::String("Eiffel Tower".to_string()));
    location.set_attribute("type".to_string(), AttributeValue::String("location".to_string()));
    location.set_attribute("coordinates".to_string(), AttributeValue::String("48.8584, 2.2945".to_string()));
    
    // Create "person" memory
    let mut person = MemoryNode::new(vec![0.5, 0.2, 0.1, 0.3]);
    person.set_attribute("title".to_string(), AttributeValue::String("John Smith".to_string()));
    person.set_attribute("type".to_string(), AttributeValue::String("person".to_string()));
    person.set_attribute("relationship".to_string(), AttributeValue::String("friend".to_string()));
    
    // Create "emotion" memory
    let mut emotion = MemoryNode::new(vec![0.8, 0.1, 0.0, 0.2]);
    emotion.set_attribute("title".to_string(), AttributeValue::String("Happiness".to_string()));
    emotion.set_attribute("type".to_string(), AttributeValue::String("emotion".to_string()));
    emotion.set_attribute("intensity".to_string(), AttributeValue::Float(0.9));
    
    // Save all memories to get their IDs
    storage.save(&event)?;
    storage.save(&location)?;
    storage.save(&person)?;
    storage.save(&emotion)?;
    
    // Store the IDs for later use
    memory_ids.insert("event".to_string(), event.id());
    memory_ids.insert("location".to_string(), location.id());
    memory_ids.insert("person".to_string(), person.id());
    memory_ids.insert("emotion".to_string(), emotion.id());
    
    // Now add connections between memories
    
    // Connect event to location
    let mut event = storage.load(event.id())?;
    event.add_connection(Connection::new(
        location.id(),
        RelationshipType::Association,
        0.8,
    ));
    
    // Connect event to person
    event.add_connection(Connection::new(
        person.id(),
        RelationshipType::Association,
        0.7,
    ));
    
    // Connect event to emotion
    event.add_connection(Connection::new(
        emotion.id(),
        RelationshipType::Causation,
        0.9,
    ));
    
    // Connect person to location
    let mut person = storage.load(person.id())?;
    person.add_connection(Connection::new(
        location.id(),
        RelationshipType::Association,
        0.6,
    ));
    
    // Save the updated memories with connections
    storage.save(&event)?;
    storage.save(&person)?;
    
    println!("Created {} connected memories", memory_ids.len());
    
    Ok(memory_ids)
}

fn traverse_graph(storage: &impl StorageEngine, start_id: Uuid, depth: usize) -> Result<(), Box<dyn Error>> {
    // Load the memory node
    let node = storage.load(start_id)?;
    
    // Print with indentation based on depth
    let indent = " ".repeat(depth * 2);
    
    // Get the title attribute or a default
    let title = node.get_attribute("title")
        .and_then(|v| if let AttributeValue::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "Untitled".to_string());
    
    // Get the type attribute or a default
    let type_attr = node.get_attribute("type")
        .and_then(|v| if let AttributeValue::String(s) = v { Some(s.clone()) } else { None })
        .unwrap_or_else(|| "Unknown".to_string());
    
    println!("{}[{}] {}", indent, type_attr, title);
    
    // If we're not too deep, follow the connections
    if depth < 3 {
        for connection in node.connections() {
            let relationship = match connection.relationship_type() {
                RelationshipType::Association => "is associated with",
                RelationshipType::Causation => "caused",
                RelationshipType::PartOf => "is part of",
                RelationshipType::Contains => "contains",
                RelationshipType::Sequence => "is followed by",
                RelationshipType::Custom(s) => s,
            };
            
            println!("{}  {} (strength: {:.2}) ->", 
                indent, relationship, connection.strength());
            
            // Recursively traverse the connected node
            traverse_graph(storage, connection.target_id(), depth + 1)?;
        }
    }
    
    Ok(())
}