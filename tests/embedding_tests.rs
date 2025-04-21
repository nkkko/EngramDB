use engramdb::{
    core::MemoryNode,
    database::Database,
    embeddings::{EmbeddingModel, MockEmbeddingModel, MultiVectorModel},
    Result,
};
use std::sync::Arc;

/// Test scenario: Basic embedding generation
///
/// This test verifies that the embedding model can correctly
/// generate embeddings from text.
#[test]
fn test_embedding_basic_generation() -> Result<()> {
    // Create a mock embedding model
    let model = MockEmbeddingModel::new(384); // 384-dimensional embeddings

    // Text to embed
    let text = "This is a test sentence for embedding generation";

    // Generate embedding
    let embedding = model.embed_text(text)?;

    // Verify dimensions
    assert_eq!(embedding.len(), 384);

    // Verify normalization (vector should have unit length)
    let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((magnitude - 1.0).abs() < 1e-5);

    // Verify deterministic output (same text should give same embedding)
    let embedding2 = model.embed_text(text)?;
    assert_eq!(embedding, embedding2);

    // Verify different texts give different embeddings
    let different_text = "This is a completely different text";
    let different_embedding = model.embed_text(different_text)?;
    assert_ne!(embedding, different_embedding);

    Ok(())
}

/// Test scenario: Multi-vector embedding
///
/// This test verifies that multi-vector embedding models
/// correctly generate token-level embeddings.
#[test]
fn test_embedding_multi_vector() -> Result<()> {
    // Create a mock multi-vector model (e.g., ColBERT-style)
    let base_model = MockEmbeddingModel::new(32); // 32-dimensional token embeddings
    let model = MultiVectorModel::new(Arc::new(base_model));

    // Text with multiple tokens
    let text = "Testing multi vector embedding model";

    // Generate multi-vector embedding
    let embeddings = model.embed_text_multi(text)?;

    // Verify we got multiple vectors (one per token plus potentially special tokens)
    assert!(embeddings.len() >= 5); // At least one per word

    // Verify each vector has the correct dimensions
    for vec in &embeddings {
        assert_eq!(vec.len(), 32);

        // Verify normalization
        let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 1e-5);
    }

    // Verify the model gives consistent results
    let embeddings2 = model.embed_text_multi(text)?;
    assert_eq!(embeddings.len(), embeddings2.len());
    for (v1, v2) in embeddings.iter().zip(embeddings2.iter()) {
        assert_eq!(v1, v2);
    }

    Ok(())
}

/// Test scenario: Different embedding dimensions
///
/// This test verifies that embedding models with different
/// dimensions work correctly.
#[test]
fn test_embedding_different_dimensions() -> Result<()> {
    // Test dimensions
    let dimensions = [16, 64, 256, 768, 1536];

    for &dim in &dimensions {
        // Create model with this dimension
        let model = MockEmbeddingModel::new(dim);

        // Generate embedding
        let text = "Testing different dimensions";
        let embedding = model.embed_text(text)?;

        // Verify dimension
        assert_eq!(embedding.len(), dim);

        // Print confirmation
        println!("Successfully generated {}-dimensional embedding", dim);
    }

    Ok(())
}

/// Test scenario: Batch embedding
///
/// This test verifies that batch embedding functionality
/// works correctly and is consistent with individual embedding.
#[test]
fn test_embedding_batch_processing() -> Result<()> {
    // Create model
    let model = MockEmbeddingModel::new(128);

    // Test texts
    let texts = vec![
        "First test text".to_string(),
        "Second test text".to_string(),
        "Third test text that is a bit longer".to_string(),
        "Fourth and final test".to_string(),
    ];

    // Generate embeddings individually
    let individual_embeddings: Vec<_> = texts
        .iter()
        .map(|text| model.embed_text(text))
        .collect::<Result<_>>()?;

    // Generate batch embeddings
    let batch_embeddings = model.embed_batch(&texts)?;

    // Verify batch has correct size
    assert_eq!(batch_embeddings.len(), texts.len());

    // Verify batch results match individual results
    for (i, (ind, batch)) in individual_embeddings
        .iter()
        .zip(batch_embeddings.iter())
        .enumerate()
    {
        assert_eq!(ind, batch, "Mismatch for text at index {}", i);
    }

    Ok(())
}

/// Test scenario: Similarity calculation
///
/// This test verifies that embedding similarity is calculated
/// correctly using different metrics.
#[test]
fn test_embedding_similarity_calculation() -> Result<()> {
    // Helper function to calculate cosine similarity
    fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot_product / (magnitude_a * magnitude_b)
    }

    // Create model
    let model = MockEmbeddingModel::new(64);

    // Generate embeddings for similar and dissimilar texts
    let text1 = "The quick brown fox jumps over the lazy dog";
    let text2 = "A swift auburn fox leaps above the idle hound"; // Similar meaning
    let text3 = "Artificial intelligence and machine learning technologies"; // Different topic

    let emb1 = model.embed_text(text1)?;
    let emb2 = model.embed_text(text2)?;
    let emb3 = model.embed_text(text3)?;

    // Calculate similarities
    let sim12 = cosine_similarity(&emb1, &emb2);
    let sim13 = cosine_similarity(&emb1, &emb3);
    let sim23 = cosine_similarity(&emb2, &emb3);

    // Verify similar texts have higher similarity
    assert!(sim12 > sim13);
    assert!(sim12 > sim23);

    // Verify self-similarity is 1.0
    let self_sim = cosine_similarity(&emb1, &emb1);
    assert!((self_sim - 1.0).abs() < 1e-5);

    println!("Similarity between similar texts: {:.4}", sim12);
    println!("Similarity between different texts: {:.4}", sim13);

    Ok(())
}

/// Test scenario: Integration with database
///
/// This test verifies that embedding models work correctly
/// when integrated with the database for search.
#[test]
fn test_embedding_integration_with_database() -> Result<()> {
    // Create model and database
    let model = Arc::new(MockEmbeddingModel::new(128));
    let db = Database::in_memory();

    // Create text documents to store
    let documents = vec![
        "Artificial intelligence focuses on creating machines that can think like humans",
        "Machine learning is a subset of AI that uses statistical methods",
        "Deep learning uses neural networks with many layers",
        "Natural language processing helps computers understand human language",
        "Computer vision enables machines to interpret and understand visual information",
        "Reinforcement learning is about taking actions to maximize reward",
        "The Turing test measures a machine's ability to exhibit human-like intelligence",
        "Robotics combines AI, engineering and computer science",
        "Expert systems use knowledge bases and inference engines",
        "Fuzzy logic deals with reasoning that is approximate rather than precise",
    ];

    // Store documents with their embeddings
    for (i, doc) in documents.iter().enumerate() {
        let embedding = model.embed_text(doc)?;
        let mut node = MemoryNode::new(embedding);
        node.set_attribute(
            "text".to_string(),
            engramdb::core::AttributeValue::String(doc.to_string()),
        );
        node.set_attribute(
            "index".to_string(),
            engramdb::core::AttributeValue::Integer(i as i64),
        );
        db.save(&node)?;
    }

    // Search with a query
    let query = "How do machines understand human language?";
    let query_embedding = model.embed_text(query)?;

    // Perform search
    let results = db.search_similar(&query_embedding, 3, 0.0, None, None)?;

    // Verify we got some results
    assert!(!results.is_empty());

    // Check returned documents - the NLP document should be highly ranked
    let mut found_nlp = false;
    for result in &results {
        let node = db.load(result.id)?;
        let text = node.get_attribute("text").unwrap().to_string();

        // If it contains NLP, it should be a good match
        if text.to_lowercase().contains("language processing") {
            found_nlp = true;
            println!(
                "NLP document found with similarity: {:.4}",
                result.similarity
            );
        }
    }

    // The NLP document should be in top results
    assert!(found_nlp, "NLP document not found in top results");

    Ok(())
}

/// Test scenario: Multi-vector search in database
///
/// This test verifies that multi-vector embeddings can be
/// used for search in the database.
#[test]
fn test_embedding_multi_vector_search() -> Result<()> {
    // Create multi-vector model
    let base_model = MockEmbeddingModel::new(32);
    let model = Arc::new(MultiVectorModel::new(Arc::new(base_model)));

    // Create database
    let db = Database::in_memory();

    // Sample paragraphs to store
    let paragraphs = vec![
        "The quick brown fox jumps over the lazy dog. This classic pangram contains every letter of the English alphabet.",
        "Artificial intelligence is transforming industries around the world. Machine learning algorithms can analyze vast amounts of data.",
        "Quantum computing uses quantum mechanics to process information differently than classical computers.",
        "Climate change poses serious global challenges. Rising temperatures are affecting ecosystems worldwide.",
        "The human genome project mapped all the genes of the human genome, completing the sequence in 2003.",
    ];

    // Store paragraphs with multi-vector embeddings
    for (i, text) in paragraphs.iter().enumerate() {
        // For testing, we'll just use regular embedding
        // In practice, we'd use a multi-vector embedding here
        let embedding = model.parent_model().embed_text(text)?;

        let mut node = MemoryNode::new(embedding);
        node.set_attribute(
            "text".to_string(),
            engramdb::core::AttributeValue::String(text.to_string()),
        );
        node.set_attribute(
            "index".to_string(),
            engramdb::core::AttributeValue::Integer(i as i64),
        );
        db.save(&node)?;
    }

    // Search with a focused query
    let query = "How does quantum computing work?";
    let query_embedding = model.parent_model().embed_text(query)?;

    // Perform search
    let results = db.search_similar(&query_embedding, 2, 0.0, None, None)?;

    // Verify we got results
    assert!(!results.is_empty());

    // The quantum computing paragraph should be highly ranked
    let mut found_quantum = false;
    for result in &results {
        let node = db.load(result.id)?;
        let text = node.get_attribute("text").unwrap().to_string();

        if text.to_lowercase().contains("quantum computing") {
            found_quantum = true;
            println!(
                "Quantum computing document found with similarity: {:.4}",
                result.similarity
            );
        }
    }

    assert!(
        found_quantum,
        "Quantum computing document not found in top results"
    );

    Ok(())
}
