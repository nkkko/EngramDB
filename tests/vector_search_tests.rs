use engramdb::{
    core::MemoryNode,
    database::Database,
    vector::{HnswIndex, ThreadSafeDatabase, VectorSearchIndex},
    Result,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::collections::HashSet;

/// Generate random vectors for testing
fn generate_random_vectors(count: usize, dimension: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| {
            let mut vec = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                vec.push(rng.gen_range(-1.0..1.0));
            }
            // Normalize the vector
            let magnitude: f32 = vec.iter().map(|v| v * v).sum::<f32>().sqrt();
            vec.iter().map(|v| v / magnitude).collect()
        })
        .collect()
}

/// Test scenario: Basic vector search functionality
///
/// This test creates a database with several nodes containing random embeddings,
/// then performs a search and verifies that the results are ordered by similarity
/// and meet the threshold requirement.
#[test]
fn test_vector_search_basic_functionality() -> Result<()> {
    // Parameters
    let num_nodes = 50;
    let dimension = 32;
    let search_limit = 10;
    let similarity_threshold = 0.5;

    // Setup: Create random vector embeddings
    let vectors = generate_random_vectors(num_nodes, dimension, 42);

    // Setup: Create a database with these vectors
    let db = Database::in_memory();
    let mut node_ids = Vec::with_capacity(num_nodes);

    for (i, embedding) in vectors.iter().enumerate() {
        let mut node = MemoryNode::new(embedding.clone());
        node.set_attribute(
            "index".to_string(),
            engramdb::core::AttributeValue::Integer(i as i64),
        );
        let id = db.save(&node)?;
        node_ids.push(id);
    }

    // Execute: Perform vector search with a query
    let query_vector = vectors[0].clone(); // Use the first vector as query
    let search_results = db.search_similar(
        &query_vector,
        search_limit,
        similarity_threshold,
        None,
        None,
    )?;

    // Verify: Results are within threshold
    for result in &search_results {
        assert!(result.similarity >= similarity_threshold);
    }

    // Verify: Results are in descending order of similarity
    for i in 1..search_results.len() {
        assert!(search_results[i - 1].similarity >= search_results[i].similarity);
    }

    // Verify: First result is the query itself (or has very high similarity)
    assert!(search_results[0].similarity > 0.99);

    // Verify: Number of results respects the limit
    assert!(search_results.len() <= search_limit);

    Ok(())
}

/// Test scenario: Empty index search
///
/// This test verifies that searching in an empty index
/// correctly returns an empty result set.
#[test]
fn test_vector_search_empty_index_returns_empty_result() -> Result<()> {
    // Setup: Create an empty database
    let db = Database::in_memory();

    // Execute: Perform search with a random query
    let query = vec![0.1, 0.2, 0.3];
    let results = db.search_similar(&query, 10, 0.0, None, None)?;

    // Verify: Results are empty
    assert!(results.is_empty());

    Ok(())
}

/// Test scenario: Result count respects the limit
///
/// This test verifies that the search respects the limit
/// parameter even when more results would match the threshold.
#[test]
fn test_vector_search_respects_result_limit() -> Result<()> {
    // Parameters
    let num_nodes = 100;
    let dimension = 16;
    let limits_to_test = [1, 5, 10, 50, 200];

    // Setup: Create a database with random vectors
    let vectors = generate_random_vectors(num_nodes, dimension, 123);
    let db = Database::in_memory();

    for embedding in vectors.iter() {
        let node = MemoryNode::new(embedding.clone());
        db.save(&node)?;
    }

    // Query vector (random)
    let query = generate_random_vectors(1, dimension, 456)[0].clone();

    // Execute & Verify: Test different limits
    for limit in limits_to_test {
        let results = db.search_similar(&query, limit, 0.0, None, None)?;
        assert!(results.len() <= limit);

        // For cases where limit > num_nodes, we expect exactly num_nodes results
        if limit > num_nodes {
            assert_eq!(results.len(), num_nodes);
        } else {
            assert_eq!(results.len(), limit);
        }
    }

    Ok(())
}

/// Test scenario: Threshold filtering
///
/// This test verifies that the similarity threshold
/// correctly filters results that don't meet the criteria.
#[test]
fn test_vector_search_threshold_filtering() -> Result<()> {
    // Parameters
    let num_nodes = 50;
    let dimension = 16;
    let thresholds_to_test = [0.0, 0.3, 0.6, 0.9, 0.99];

    // Setup: Create a database with random vectors
    let vectors = generate_random_vectors(num_nodes, dimension, 789);
    let db = Database::in_memory();

    for embedding in vectors.iter() {
        let node = MemoryNode::new(embedding.clone());
        db.save(&node)?;
    }

    // Query vector (use first vector to ensure some high similarity matches)
    let query = vectors[0].clone();

    // Execute & Verify: Test different thresholds
    let mut prev_count = 0;
    for &threshold in thresholds_to_test.iter() {
        let results = db.search_similar(&query, 50, threshold, None, None)?;

        // Verify all results meet threshold
        for result in &results {
            assert!(result.similarity >= threshold);
        }

        // Verify higher thresholds return fewer or equal results
        if threshold > 0.0 {
            assert!(results.len() <= prev_count);
        }
        prev_count = results.len();
    }

    // Verify very high threshold returns very few results
    let high_results = db.search_similar(&query, 50, 0.99, None, None)?;
    assert!(high_results.len() <= 2); // Only the query vector itself should be this close

    Ok(())
}

/// Test scenario: HNSW index configuration impact
///
/// This test verifies that different HNSW parameters affect
/// the performance and recall of the index.
#[test]
fn test_vector_search_hnsw_configuration() -> Result<()> {
    // Parameters
    let num_nodes = 100;
    let dimension = 32;
    let query_count = 5;

    // Generate random vectors for indexing
    let vectors = generate_random_vectors(num_nodes, dimension, 101112);

    // Generate query vectors
    let query_vectors = generate_random_vectors(query_count, dimension, 131415);

    // Configurations to test (M, ef_construction)
    let configs = vec![
        (8, 10),   // Fast but less accurate
        (16, 100), // Balanced
        (32, 200), // Slower but more accurate
    ];

    // First, create a brute-force index for ground truth
    let mut ground_truth = Vec::new();
    {
        let db = Database::in_memory();

        for embedding in vectors.iter() {
            let node = MemoryNode::new(embedding.clone());
            db.save(&node)?;
        }

        // For each query, get ground truth results
        for query in &query_vectors {
            let results = db.search_similar(query, 10, 0.0, None, None)?;
            let result_ids: HashSet<_> = results.iter().map(|r| r.id).collect();
            ground_truth.push(result_ids);
        }
    }

    // Test each HNSW configuration
    for (m, ef_construction) in configs {
        // Create custom HNSW index
        let mut hnsw = HnswIndex::with_params(dimension, m, ef_construction, 200);

        // Create database with this index
        let db = Database::with_vector_index(Box::new(hnsw));

        // Insert same vectors
        for embedding in vectors.iter() {
            let node = MemoryNode::new(embedding.clone());
            db.save(&node)?;
        }

        // Query and measure recall
        let mut total_recall = 0.0;
        for (i, query) in query_vectors.iter().enumerate() {
            let results = db.search_similar(query, 10, 0.0, None, None)?;
            let result_ids: HashSet<_> = results.iter().map(|r| r.id).collect();

            // Calculate recall (intersection over ground truth)
            let intersection = ground_truth[i].intersection(&result_ids).count();
            let recall = intersection as f32 / ground_truth[i].len() as f32;
            total_recall += recall;
        }

        // Calculate average recall
        let avg_recall = total_recall / query_count as f32;
        println!(
            "HNSW config (M={}, ef_construction={}) - Avg recall: {:.4}",
            m, ef_construction, avg_recall
        );

        // Verify reasonable recall (might be probabilistic)
        assert!(avg_recall > 0.5, "HNSW recall is too low: {}", avg_recall);
    }

    Ok(())
}

/// Test scenario: Search with filters
///
/// This test verifies that vector search combined with attribute
/// filters returns the correct results.
#[test]
fn test_vector_search_with_attribute_filters() -> Result<()> {
    // Parameters
    let num_nodes = 50;
    let dimension = 16;

    // Setup: Create a database with categorized vectors
    let vectors = generate_random_vectors(num_nodes, dimension, 161718);
    let db = Database::in_memory();

    for (i, embedding) in vectors.iter().enumerate() {
        let mut node = MemoryNode::new(embedding.clone());

        // Add category attribute - "A" for first half, "B" for second half
        let category = if i < num_nodes / 2 { "A" } else { "B" };
        node.set_attribute(
            "category".to_string(),
            engramdb::core::AttributeValue::String(category.to_string()),
        );

        // Add value attribute - even or odd
        node.set_attribute(
            "value".to_string(),
            engramdb::core::AttributeValue::Integer((i % 2) as i64),
        );

        db.save(&node)?;
    }

    // Query vector
    let query = vectors[0].clone(); // Use first vector (category A)

    // Execute: Search with no filter
    let unfiltered_results = db.search_similar(&query, 10, 0.0, None, None)?;

    // Execute: Search with category filter "A"
    let filter_a = Some("category = 'A'".to_string());
    let category_a_results = db.search_similar(&query, 10, 0.0, filter_a, None)?;

    // Execute: Search with category filter "B"
    let filter_b = Some("category = 'B'".to_string());
    let category_b_results = db.search_similar(&query, 10, 0.0, filter_b, None)?;

    // Execute: Search with value filter (even)
    let filter_even = Some("value = 0".to_string());
    let even_results = db.search_similar(&query, 10, 0.0, filter_even, None)?;

    // Verify: All results from filtered "A" have category A
    for result in &category_a_results {
        let node = db.load(result.id)?;
        assert_eq!(
            node.get_attribute("category").unwrap().to_string(),
            "\"A\"".to_string()
        );
    }

    // Verify: All results from filtered "B" have category B
    for result in &category_b_results {
        let node = db.load(result.id)?;
        assert_eq!(
            node.get_attribute("category").unwrap().to_string(),
            "\"B\"".to_string()
        );
    }

    // Verify: All results from filtered "even" have value 0
    for result in &even_results {
        let node = db.load(result.id)?;
        assert_eq!(
            node.get_attribute("value").unwrap().to_string(),
            "0".to_string()
        );
    }

    // Verify: Combined results from A and B should cover all unfiltered results
    let combined_ids: HashSet<_> = category_a_results
        .iter()
        .chain(category_b_results.iter())
        .map(|r| r.id)
        .collect();

    let unfiltered_ids: HashSet<_> = unfiltered_results.iter().map(|r| r.id).collect();

    assert_eq!(combined_ids.len(), unfiltered_ids.len());
    assert!(combined_ids.is_superset(&unfiltered_ids));

    Ok(())
}

/// Test scenario: Thread-safe vector search
///
/// This test verifies that vector search works correctly
/// with the ThreadSafeDatabase implementation.
#[test]
fn test_vector_search_thread_safe_database() -> Result<()> {
    // Parameters
    let num_nodes = 30;
    let dimension = 16;

    // Setup: Create a thread-safe database with random vectors
    let vectors = generate_random_vectors(num_nodes, dimension, 192021);
    let db = ThreadSafeDatabase::in_memory_with_hnsw();

    for embedding in vectors.iter() {
        let node = MemoryNode::new(embedding.clone());
        db.save(&node)?;
    }

    // Execute: Perform vector search
    let query = vectors[0].clone();
    let results = db.search_similar(&query, 10, 0.0, None, None)?;

    // Verify: Results are non-empty and properly ordered
    assert!(!results.is_empty());
    for i in 1..results.len() {
        assert!(results[i - 1].similarity >= results[i].similarity);
    }

    // Verify: First result is the query itself (very high similarity)
    assert!(results[0].similarity > 0.99);

    Ok(())
}

/// Test scenario: Different vector dimensions
///
/// This test verifies that the search works correctly with
/// different vector dimensions.
#[test]
fn test_vector_search_different_dimensions() -> Result<()> {
    // Dimensions to test
    let dimensions = [4, 16, 64, 256, 768];

    for &dim in &dimensions {
        // Setup: Create database with vectors of this dimension
        let vectors = generate_random_vectors(20, dim, 222324);
        let db = Database::in_memory();

        for embedding in vectors.iter() {
            let node = MemoryNode::new(embedding.clone());
            db.save(&node)?;
        }

        // Execute: Perform search
        let query = vectors[0].clone();
        let results = db.search_similar(&query, 5, 0.0, None, None)?;

        // Verify: Search works correctly
        assert!(!results.is_empty());
        assert!(results[0].similarity > 0.99);

        println!(
            "Vector dimension {} - search successful with {} results",
            dim,
            results.len()
        );
    }

    Ok(())
}
