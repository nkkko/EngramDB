use engramdb::{Database, MemoryNode};
use engramdb::core::AttributeValue;
use engramdb::embeddings::EmbeddingService;
use std::error::Error;

fn main() -> Result<(), Box<dyn Error>> {
    println!("EngramDB with HNSW Index for Agent Memory");
    println!("------------------------------------------");
    
    // Create a database with HNSW vector index for faster search
    let mut db = Database::in_memory_with_hnsw();
    println!("Created database with HNSW vector index");
    
    // Initialize the embedding service for text-to-vector conversion
    // This will use a mock provider for the demonstration
    let embedding_service = EmbeddingService::new_mock(384);
    println!("Initialized embedding service");
    
    // Create some memories from text
    let memories = vec![
        "The agent discussed project planning with the user",
        "The user asked about their calendar for next week",
        "The agent helped the user draft an email to a colleague",
        "The user requested information about machine learning concepts",
        "The agent provided a summary of recent news articles",
        "The user asked for help with coding a Python function",
        "The agent explained how to use a REST API",
        "The user wanted to know about climate change statistics",
        "The agent recommended a book on productivity",
        "The user asked about the weather forecast for the weekend",
        // Add more memories here
    ];
    
    println!("\nCreating {} memories from text...", memories.len());
    
    // Store memories in the database
    for (i, text) in memories.iter().enumerate() {
        // Create a memory node from text
        let mut node = MemoryNode::new(embedding_service.generate_for_document(text, None)?);
        
        // Add attributes
        node.set_attribute("content".to_string(), AttributeValue::String(text.to_string()));
        node.set_attribute("type".to_string(), AttributeValue::String("conversation".to_string()));
        node.set_attribute("importance".to_string(), AttributeValue::Float(0.7));
        node.set_attribute("timestamp".to_string(), AttributeValue::Integer(
            chrono::Utc::now().timestamp() - i as i64 * 3600 // Memories from different times
        ));
        
        // Save to database
        let id = db.save(&node)?;
        println!("  Created memory {}: {}", i + 1, id);
    }
    
    // Perform semantic search using the HNSW index
    println!("\nPerforming semantic searches...");
    
    let queries = vec![
        "What did the user ask about scheduling?",
        "Tell me about coding discussions",
        "Any conversations about news or current events?",
    ];
    
    for query in queries {
        println!("\nQuery: '{}'", query);
        
        // Generate embeddings for the query
        let query_embedding = embedding_service.generate_for_query(query)?;
        
        // Search for similar memories
        let results = db.search_similar(&query_embedding, 2, 0.0)?;
        
        // Display results
        println!("Found {} results:", results.len());
        for (i, (memory_id, similarity)) in results.iter().enumerate() {
            // Load the memory to get its content
            let memory = db.load(*memory_id)?;
            
            if let Some(AttributeValue::String(content)) = memory.get_attribute("content") {
                println!("  Result {}: (similarity: {:.4})", i + 1, similarity);
                println!("    Content: '{}'", content);
            }
        }
    }
    
    println!("\nHNSW vector search provides significantly faster retrieval than linear search,");
    println!("especially as the database grows. This is important for real-time agent interactions.");
    
    Ok(())
}