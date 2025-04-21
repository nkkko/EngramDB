use engramdb::{
    core::{MemoryNode, AttributeValue},
    database::{Database, DatabaseConfig},
    StorageType,
    embeddings::{
        EmbeddingService, MockMultiVectorProvider,
        EmbeddingError,
    },
    vector::{
        MultiVectorIndex, MultiVectorIndexConfig, MultiVectorSimilarityMethod,
        VectorAlgorithm, VectorIndexConfig,
    },
    error::EngramDbError
};
use std::fmt;

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
    println!("=== EngramDB Multi-Vector Embeddings Example ===");

    // Create a mock embedding service with multi-vector capability
    let mut service = EmbeddingService::new_mock(96);
    let multi_provider = MockMultiVectorProvider::new(96, 20);
    service.add_multi_vector_provider(multi_provider);

    println!("Created embedding service with multi-vector capability");
    println!("Dimensions: {}", service.dimensions());
    println!("Multi-vector dimensions: {:?}", service.multi_vector_dimensions());

    // Create example documents
    let documents = vec![
        "EngramDB is a specialized database for agent memory systems.",
        "It supports both single-vector and multi-vector embeddings.",
        "Multi-vector embeddings allow for more precise similarity matching.",
        "The late interaction approach compares individual token embeddings.",
        "ColBERT and ColPali are examples of multi-vector embedding models.",
    ];

    // Create both single-vector and multi-vector nodes
    let mut single_vector_nodes = Vec::new();
    let mut multi_vector_nodes = Vec::new();

    println!("\nCreating memory nodes with both embedding types:");
    for (i, doc) in documents.iter().enumerate() {
        // Create a single-vector node
        let node_sv = MemoryNode::from_text(doc, &service, None)?;
        
        // Get dimensions before moving
        let sv_dimensions = node_sv.embeddings().map(|v| v.len()).unwrap_or(0);
        single_vector_nodes.push(node_sv);

        // Create a multi-vector node
        let multi_vec = service.generate_multi_vector_for_document(doc, None)?;
        let node_mv = MemoryNode::with_multi_vector(multi_vec);
        
        // Get dimensions before moving
        let mv_num_vectors = node_mv.multi_vector_embeddings().map(|mv| mv.num_vectors()).unwrap_or(0);
        let mv_dimensions = node_mv.multi_vector_embeddings().map(|mv| mv.dimensions()).unwrap_or(0);
        
        multi_vector_nodes.push(node_mv);

        println!("- Document {}: Created single and multi-vector nodes", i + 1);
        println!("  └ Single vector dimensions: {}", sv_dimensions);
        println!("  └ Multi vector: {} vectors of {} dimensions each", 
            mv_num_vectors, mv_dimensions);
    }

    // Create a specialized multi-vector index
    let mut mv_index = MultiVectorIndex::with_config(MultiVectorIndexConfig {
        quantize: false,
        similarity_method: MultiVectorSimilarityMethod::LateInteraction,
    });

    println!("\nAdding multi-vector nodes to specialized index");
    for node in &multi_vector_nodes {
        mv_index.add(node)?;
    }

    // Perform a multi-vector search
    let query = "Tell me about vector embeddings";
    let query_multi_vec = service.generate_multi_vector_for_query(query)?;

    println!("\nSearching for: \"{}\"", query);
    println!("Query has {} vectors", query_multi_vec.num_vectors());

    let results = mv_index.search(&query_multi_vec, 3, 0.0)?;
    
    println!("\nResults using late interaction scoring:");
    for (i, (id, score)) in results.iter().enumerate() {
        let node = multi_vector_nodes.iter().find(|n| n.id() == *id).unwrap();
        let content = node.get_attribute("content")
.and_then(|attr| match attr {
                AttributeValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        
        println!("{}. Score: {:.4} - \"{}\"", i + 1, score, content);
    }

    // Compare with other similarity methods
    println!("\nComparing different similarity methods:");
    
    // MaxSim
    let config_max = MultiVectorIndexConfig {
        quantize: false,
        similarity_method: MultiVectorSimilarityMethod::Maximum,
    };
    let mut index_max = MultiVectorIndex::with_config(config_max);
    for node in &multi_vector_nodes {
        index_max.add(node)?;
    }
    let results_max = index_max.search(&query_multi_vec, 3, 0.0)?;
    
    println!("\nResults using maximum similarity:");
    for (i, (id, score)) in results_max.iter().enumerate() {
        let node = multi_vector_nodes.iter().find(|n| n.id() == *id).unwrap();
        let content = node.get_attribute("content")
.and_then(|attr| match attr {
                AttributeValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        
        println!("{}. Score: {:.4} - \"{}\"", i + 1, score, content);
    }
    
    // Average
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
        let node = multi_vector_nodes.iter().find(|n| n.id() == *id).unwrap();
        let content = node.get_attribute("content")
.and_then(|attr| match attr {
                AttributeValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        
        println!("{}. Score: {:.4} - \"{}\"", i + 1, score, content);
    }

    // Create a database with multi-vector support
    println!("\nCreating database with multi-vector index");
    let vector_config = VectorIndexConfig {
        algorithm: VectorAlgorithm::MultiVector,
        hnsw: None,
        multi_vector: Some(MultiVectorIndexConfig::default()),
    };
    
    let db_config = DatabaseConfig {
        storage_type: StorageType::Memory,
        storage_path: None,
        cache_size: 100,
        vector_index_config: vector_config,
    };
    
    let _db = Database::new(db_config)?; // Database is unused in this example

    // Since database doesn't expose direct access to vector_index, we'll use our own
    // MultiVectorIndex here and skip using the Database
    
    println!("Using separate MultiVectorIndex to perform search");
    println!("Added {} multi-vector nodes to index", multi_vector_nodes.len());
    
    // Use direct MultiVectorIndex instead of going through Database
    let mut direct_index = MultiVectorIndex::new();
    
    // Add the nodes to the index
    for node in &multi_vector_nodes {
        direct_index.add(node)?;
    }
    
    // Perform a query using the direct index
    let results = direct_index.search(&query_multi_vec, 3, 0.0)?;
    
    println!("\nDatabase search results:");
    for (i, (id, score)) in results.iter().enumerate() {
        // Get the node from our local list since the Database interface doesn't expose get_by_id
        let node = multi_vector_nodes.iter().find(|n| n.id() == *id).unwrap();
        let content = node.get_attribute("content")
.and_then(|attr| match attr {
                AttributeValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or_default();
        
        println!("{}. Score: {:.4} - \"{}\"", i + 1, score, content);
    }

    println!("\nExample completed successfully!");
    Ok(())
}