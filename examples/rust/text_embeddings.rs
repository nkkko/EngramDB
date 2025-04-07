//! Example demonstrating text embedding functionality in EngramDB.
//!
//! This example shows how to:
//! 1. Create an embedding service to convert text to vectors
//! 2. Create memories from text content
//! 3. Use these memories for semantic search

use engramdb::{
    core::{AttributeValue, MemoryNode},
    database::Database,
    embeddings::EmbeddingService,
    vector::similarity::cosine_similarity,
};
use std::collections::HashMap;
use std::path::PathBuf;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logger
    env_logger::init();
    
    println!("EngramDB Text Embeddings Example");
    println!("--------------------------------");
    
    // Create a database for our examples
    let db_path = PathBuf::from("examples_text_embeddings_rust.engramdb");
    let mut db = Database::single_file(&db_path)?;
    println!("Database created at {:?}", db_path);
    
    // Initialize the embedding service (will use mock embeddings if ML deps not installed)
    let embedding_service = EmbeddingService::default();
    println!("Embedding service initialized (dimensions: {})", embedding_service.dimensions());
    
    // Create memory nodes from text
    let texts = vec![
        "Artificial intelligence is transforming how we interact with technology",
        "Machine learning algorithms can identify patterns in large datasets",
        "Natural language processing enables computers to understand human speech",
        "Computer vision systems can recognize objects in images and videos",
        "Reinforcement learning helps AI agents learn through trial and error",
        "Quantum computing may accelerate certain AI algorithms exponentially",
    ];
    
    // Store all texts as memories
    println!("\nCreating memories from text...");
    
    let mut memory_ids = Vec::new();
    
    for (i, &text) in texts.iter().enumerate() {
        // Create additional attributes
        let mut attributes = HashMap::new();
        attributes.insert("title".to_string(), AttributeValue::String(format!("AI Concept {}", i+1)));
        
        // The create_memory_from_text method handles everything:
        // 1. Converts text to embeddings
        // 2. Creates a memory node with those embeddings
        // 3. Stores the text in the 'content' attribute
        // 4. Saves the memory to the database
        let memory_id = db.create_memory_from_text(
            text,
            &embedding_service,
            Some("AI concepts"),
            Some(&attributes),
        )?;
        
        memory_ids.push(memory_id);
        println!("Created memory {}: ID {}", i+1, memory_id);
    }
    
    // Demonstrate semantic search
    println!("\nPerforming semantic searches...");
    
    // Create query embeddings
    let queries = vec![
        "How is AI changing our technology interactions?",
        "Finding patterns in data",
        "Understanding human language",
        "Recognizing objects in pictures",
        "Learning through experimentation",
        "Advanced computing methods",
    ];
    
    for (i, query) in queries.iter().enumerate() {
        println!("\nQuery: '{}'", query);
        
        // Generate embeddings for the query
        let query_embedding = match embedding_service.generate_for_query(query) {
            Ok(emb) => emb,
            Err(e) => {
                println!("Error generating embeddings: {}", e);
                continue;
            }
        };
        
        // Search for similar memories
        let results = db.search_similar(&query_embedding, 2, 0.0, None, None)?;
        
        // Display results
        println!("Found {} results:", results.len());
        for (j, (memory_id, similarity)) in results.iter().enumerate() {
            // Load the memory to get its content
            let memory = db.load(*memory_id)?;
            let content = memory.get_attribute("content")
                .and_then(|attr| if let AttributeValue::String(s) = attr { Some(s.as_str()) } else { None })
                .unwrap_or("No content");
                
            println!("  Result {}: (similarity: {:.4})", j+1, similarity);
            println!("    Content: '{}'", content);
        }
    }
    
    println!("\nExample completed successfully!");
    
    Ok(())
}