use engramdb::{
    core::{MemoryNode, AttributeValue},
    database::{Database, DatabaseConfig},
    StorageType,
    embeddings::{
        EmbeddingService, EmbeddingModel, EmbeddingError
    },
    vector::{
        MultiVectorIndex, MultiVectorIndexConfig, MultiVectorSimilarityMethod,
        VectorAlgorithm, VectorIndexConfig,
    },
    error::EngramDbError
};
use std::fmt;
use std::time::Instant;

// Custom error type that can handle both EngramDbError and EmbeddingError
#[derive(Debug)]
enum ExampleError {
    DatabaseError(EngramDbError),
    EmbeddingError(EmbeddingError),
}

impl fmt::Display for ExampleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExampleError::DatabaseError(e) => write!(f, "Database error: {}", e),
            ExampleError::EmbeddingError(e) => write!(f, "Embedding error: {}", e),
        }
    }
}

impl std::error::Error for ExampleError {}

impl From<EngramDbError> for ExampleError {
    fn from(err: EngramDbError) -> Self {
        ExampleError::DatabaseError(err)
    }
}

impl From<EmbeddingError> for ExampleError {
    fn from(err: EmbeddingError) -> Self {
        ExampleError::EmbeddingError(err)
    }
}

impl From<Box<dyn std::error::Error>> for ExampleError {
    fn from(err: Box<dyn std::error::Error>) -> Self {
        ExampleError::EmbeddingError(EmbeddingError::GenerationFailed(err.to_string()))
    }
}

fn main() -> std::result::Result<(), ExampleError> {
    println!("=== EngramDB Jina ColBERT Multi-Vector Example ===");

    // Initialize the embedding service with Jina ColBERT v2 model
    println!("Initializing embedding service with Jina ColBERT v2 model...");
    
    // In production, use the real model. For now, we'll use a mock to make the example run without Python.
    #[cfg(feature = "python")]
    let service = EmbeddingService::with_multi_vector_model(EmbeddingModel::JinaColBERTv2);
    
    #[cfg(not(feature = "python"))]
    let service = EmbeddingService::new_mock_multi_vector(768, 32);

    println!("Initialized ColBERT service:");
    println!("  Single-vector dimensions: {}", service.dimensions());
    if let Some(dim) = service.multi_vector_dimensions() {
        println!("  Multi-vector dimensions: {}", dim);
    }
    println!("  Has multi-vector capability: {}", service.has_multi_vector());

    // Example documents for the knowledge base
    let documents = vec![
        "EngramDB is a specialized database for agent memory systems.",
        "It supports both single-vector and multi-vector embeddings.",
        "Multi-vector embeddings allow for more precise similarity matching.",
        "The late interaction approach compares individual token embeddings.",
        "ColBERT and ColPali are examples of multi-vector embedding models.",
    ];

    // Create nodes with multi-vector embeddings
    println!("\nCreating memory nodes with ColBERT multi-vector embeddings...");
    let mut multi_vector_nodes = Vec::new();

    for (i, &doc) in documents.iter().enumerate() {
        let start_time = Instant::now();

        // Generate multi-vector embeddings
        let multi_vec = service.generate_multi_vector_for_document(doc, None)?;

        // Create a memory node with multi-vector embeddings
        let mut node = MemoryNode::with_multi_vector(multi_vec.clone());
        node.set_attribute("content".to_string(), AttributeValue::String(doc.to_string()));
        
        let gen_time = start_time.elapsed();
        
        println!("  Document {}: Created multi-vector node in {:.2?}", i + 1, gen_time);
        println!("    - {} token vectors of {} dimensions each", 
                 multi_vec.num_vectors(), multi_vec.dimensions());
        println!("    - Content: \"{}\"", doc);
        
        multi_vector_nodes.push(node);
    }

    // Create a specialized multi-vector index for search
    println!("\nCreating multi-vector index...");
    let mut mv_index = MultiVectorIndex::with_config(MultiVectorIndexConfig {
        quantize: false,
        similarity_method: MultiVectorSimilarityMethod::LateInteraction,
    });

    // Add the nodes to the index
    for node in &multi_vector_nodes {
        mv_index.add(node)?;
    }
    
    println!("Added {} nodes to index", multi_vector_nodes.len());

    // Perform queries
    let queries = vec![
        "What are multi-vector embeddings?",
        "Tell me about the late interaction approach",
        "What is EngramDB used for?",
    ];

    println!("\nPerforming ColBERT multi-vector searches:");

    for query in &queries {
        println!("\nQuery: \"{}\"", query);

        // Generate multi-vector query embeddings
        let start_time = Instant::now();
        let query_multi_vec = service.generate_multi_vector_for_query(query)?;
        let gen_time = start_time.elapsed();

        println!("  Generated query embedding with {} token vectors in {:.2?}", 
                 query_multi_vec.num_vectors(), gen_time);

        // Perform search
        let start_time = Instant::now();
        let results = mv_index.search(&query_multi_vec, 3, 0.0)?;
        let search_time = start_time.elapsed();

        println!("  Search completed in {:.2?}, found {} results:", 
                 search_time, results.len());

        for (i, (id, score)) in results.iter().enumerate() {
            // Find the node with matching ID
            if let Some(node) = multi_vector_nodes.iter().find(|n| n.id() == *id) {
                let content = node.get_attribute("content")
                    .and_then(|attr| match attr {
                        AttributeValue::String(s) => Some(s.as_str()),
                        _ => None,
                    })
                    .unwrap_or_default();
                
                println!("    {}. Score: {:.4} - \"{}\"", i + 1, score, content);
            }
        }
    }

    // Also demonstrate with different similarity methods
    println!("\nComparing different similarity methods:");
    
    // Test with MaxSim
    let config_max = MultiVectorIndexConfig {
        quantize: false,
        similarity_method: MultiVectorSimilarityMethod::Maximum,
    };
    let mut index_max = MultiVectorIndex::with_config(config_max);
    for node in &multi_vector_nodes {
        index_max.add(node)?;
    }
    
    // Generate embeddings for comparison query
    let comparison_query = "What are the benefits of multi-vector embeddings?";
    let query_multi_vec = service.generate_multi_vector_for_query(comparison_query)?;
    
    println!("\nQuery: \"{}\"", comparison_query);
    
    // Maximum similarity
    let results_max = index_max.search(&query_multi_vec, 3, 0.0)?;
    println!("\nResults using maximum similarity:");
    for (i, (id, score)) in results_max.iter().enumerate() {
        if let Some(node) = multi_vector_nodes.iter().find(|n| n.id() == *id) {
            let content = node.get_attribute("content")
                .and_then(|attr| match attr {
                    AttributeValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or_default();
            
            println!("  {}. Score: {:.4} - \"{}\"", i + 1, score, content);
        }
    }
    
    // Average similarity
    let config_avg = MultiVectorIndexConfig {
        quantize: false,
        similarity_method: MultiVectorSimilarityMethod::Average,
    };
    let mut index_avg = MultiVectorIndex::with_config(config_avg);
    for node in &multi_vector_nodes {
        index_avg.add(node)?;
    }
    let results_avg = index_avg.search(&query_multi_vec, 3, 0.0)?;
    
    println!("\nResults using average similarity:");
    for (i, (id, score)) in results_avg.iter().enumerate() {
        if let Some(node) = multi_vector_nodes.iter().find(|n| n.id() == *id) {
            let content = node.get_attribute("content")
                .and_then(|attr| match attr {
                    AttributeValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or_default();
            
            println!("  {}. Score: {:.4} - \"{}\"", i + 1, score, content);
        }
    }
    
    // Late interaction (ColBERT-style)
    let config_late = MultiVectorIndexConfig {
        quantize: false,
        similarity_method: MultiVectorSimilarityMethod::LateInteraction,
    };
    let mut index_late = MultiVectorIndex::with_config(config_late);
    for node in &multi_vector_nodes {
        index_late.add(node)?;
    }
    let results_late = index_late.search(&query_multi_vec, 3, 0.0)?;
    
    println!("\nResults using late interaction similarity (ColBERT-style):");
    for (i, (id, score)) in results_late.iter().enumerate() {
        if let Some(node) = multi_vector_nodes.iter().find(|n| n.id() == *id) {
            let content = node.get_attribute("content")
                .and_then(|attr| match attr {
                    AttributeValue::String(s) => Some(s.as_str()),
                    _ => None,
                })
                .unwrap_or_default();
            
            println!("  {}. Score: {:.4} - \"{}\"", i + 1, score, content);
        }
    }

    println!("\nExample completed successfully!");
    Ok(())
}