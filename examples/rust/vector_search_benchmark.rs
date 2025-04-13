use engramdb::{Database, MemoryNode};
use engramdb::vector::{VectorAlgorithm, VectorIndexConfig, HnswConfig};
use std::time::{Duration, Instant};
use rand::prelude::*;

fn main() {
    // Parameters for the benchmark
    let vector_dim = 384;     // Typical embedding dimension
    let dataset_size = 10000; // Number of vectors in the dataset
    let num_queries = 100;    // Number of search queries to perform
    let top_k = 10;           // Number of results to retrieve per query
    
    println!("EngramDB Vector Search Benchmark");
    println!("--------------------------------");
    println!("Dataset size: {} vectors of dimension {}", dataset_size, vector_dim);
    println!("Queries: {} with top_k = {}", num_queries, top_k);
    println!();
    
    // Generate random data
    println!("Generating random data...");
    let (nodes, queries) = generate_random_data(vector_dim, dataset_size, num_queries);
    
    // Benchmark linear search
    println!("\nBenchmarking Linear Search:");
    let linear_db = Database::in_memory();
    benchmark_search(&linear_db, &nodes, &queries, top_k);
    
    // Benchmark HNSW search with default parameters
    println!("\nBenchmarking HNSW Search (default parameters):");
    let hnsw_db = Database::in_memory_with_hnsw();
    benchmark_search(&hnsw_db, &nodes, &queries, top_k);
    
    // Benchmark HNSW search with optimized parameters
    println!("\nBenchmarking HNSW Search (optimized parameters):");
    let config = engramdb::DatabaseConfig {
        storage_type: engramdb::StorageType::Memory,
        storage_path: None,
        cache_size: 100,
        vector_index_config: VectorIndexConfig {
            algorithm: VectorAlgorithm::HNSW,
            hnsw: Some(HnswConfig {
                m: 32,                  // More connections per node
                ef_construction: 200,   // More neighbors during construction
                ef: 50,                 // More candidates during search
                level_multiplier: 1.0 / std::f32::consts::LN_2,
                max_level: 16,
            }),
        },
    };
    let hnsw_optimized_db = Database::new(config).unwrap();
    benchmark_search(&hnsw_optimized_db, &nodes, &queries, top_k);
}

// Generate random normalized vectors for benchmarking
fn generate_random_data(
    dimensions: usize,
    dataset_size: usize,
    num_queries: usize,
) -> (Vec<MemoryNode>, Vec<Vec<f32>>) {
    let mut rng = rand::thread_rng();
    
    // Generate dataset
    let mut nodes = Vec::with_capacity(dataset_size);
    for _ in 0..dataset_size {
        let vec = generate_random_vector(&mut rng, dimensions);
        nodes.push(MemoryNode::new(vec));
    }
    
    // Generate queries
    let mut queries = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        queries.push(generate_random_vector(&mut rng, dimensions));
    }
    
    (nodes, queries)
}

// Generate a random normalized vector
fn generate_random_vector(rng: &mut ThreadRng, dimensions: usize) -> Vec<f32> {
    let mut vec = Vec::with_capacity(dimensions);
    
    // Generate random values
    for _ in 0..dimensions {
        vec.push(rng.gen::<f32>() * 2.0 - 1.0); // Values between -1 and 1
    }
    
    // Normalize the vector
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
    for x in &mut vec {
        *x /= norm;
    }
    
    vec
}

// Benchmark search performance
fn benchmark_search(
    db: &Database,
    nodes: &[MemoryNode],
    queries: &[Vec<f32>],
    top_k: usize,
) {
    // First, add all nodes to the database
    let start_time = Instant::now();
    
    for node in nodes {
        db.save(node).unwrap();
    }
    
    let indexing_time = start_time.elapsed();
    println!("  Index build time: {:?}", indexing_time);
    
    // Perform searches
    let mut total_search_time = Duration::new(0, 0);
    let mut total_results = 0;
    
    for query in queries {
        let search_start = Instant::now();
        let results = db.search_similar(query, top_k, 0.0).unwrap();
        total_search_time += search_start.elapsed();
        total_results += results.len();
    }
    
    let avg_search_time = total_search_time.as_secs_f64() / queries.len() as f64;
    let searches_per_second = 1.0 / avg_search_time;
    
    println!("  Average results per query: {:.2}", total_results as f64 / queries.len() as f64);
    println!("  Average search time: {:.6} seconds", avg_search_time);
    println!("  Searches per second: {:.2}", searches_per_second);
    println!("  Total search time: {:?}", total_search_time);
}