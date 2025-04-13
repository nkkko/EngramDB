use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use engramdb::core::MemoryNode;
use engramdb::vector::{VectorIndex, HnswIndex, create_vector_index, VectorIndexConfig, VectorAlgorithm};

fn create_random_embedding(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::random::<f32>()).collect()
}

// Create a function to measure memory usage with large embedding vectors
fn benchmark_large_embeddings_memory(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_embeddings_memory");
    
    // Test with different embedding sizes to see memory impact
    for &dim in &[128, 768, 1536] {  // Using smaller set for faster benchmarking
        // Test both VectorIndex and HnswIndex to compare memory usage
        let linear_id = BenchmarkId::new("linear", dim);
        let hnsw_id = BenchmarkId::new("hnsw", dim);
        
        // Generate a consistent set of nodes for fair comparison
        let node_count = 100;  // Reduce count for faster benchmarking
        let nodes: Vec<_> = (0..node_count)
            .map(|_| MemoryNode::new(create_random_embedding(dim)))
            .collect();
        
        // Benchmark adding large vectors to linear index
        group.bench_with_input(linear_id, &nodes, |b, nodes| {
            b.iter(|| {
                let mut index = VectorIndex::new();
                for node in nodes {
                    index.add(black_box(node)).unwrap();
                }
                black_box(&index);
            });
        });
        
        // Benchmark adding large vectors to HNSW index
        group.bench_with_input(hnsw_id, &nodes, |b, nodes| {
            b.iter(|| {
                // Create an HNSW index using the trait implementation
                let config = VectorIndexConfig {
                    algorithm: VectorAlgorithm::HNSW,
                    hnsw: None,
                };
                let mut index = create_vector_index(&config);
                
                for node in nodes {
                    index.add(black_box(node)).unwrap();
                }
                black_box(&index);
            });
        });
    }
    
    group.finish();
}

// Benchmark search performance on large vectors
fn benchmark_large_embeddings_search(c: &mut Criterion) {
    let mut group = c.benchmark_group("large_embeddings_search");
    
    // Test with moderate embedding size but varying dataset size
    let dim = 768; // Common embedding size for language models
    
    for &size in &[100, 1000] {  // Reduced for faster benchmarking
        // Create indices through the trait interface
        let mut linear_index = VectorIndex::new();
        
        // Create HNSW index via config
        let hnsw_config = VectorIndexConfig {
            algorithm: VectorAlgorithm::HNSW,
            hnsw: None,
        };
        let mut hnsw_index = create_vector_index(&hnsw_config);
        
        // Add random embeddings to both indices
        for _ in 0..size {
            let node = MemoryNode::new(create_random_embedding(dim));
            linear_index.add(&node).unwrap();
            hnsw_index.add(&node).unwrap();
        }
        
        // Create a random query vector
        let query = create_random_embedding(dim);
        
        // Benchmark linear search
        let linear_id = BenchmarkId::new("linear_search", size);
        group.bench_function(linear_id, |b| {
            b.iter(|| {
                let results = linear_index
                    .search(black_box(&query), black_box(10), black_box(0.0))
                    .unwrap();
                black_box(results)
            })
        });
        
        // Benchmark HNSW search
        let hnsw_id = BenchmarkId::new("hnsw_search", size);
        group.bench_function(hnsw_id, |b| {
            b.iter(|| {
                let results = hnsw_index
                    .search(black_box(&query), black_box(10), black_box(0.0))
                    .unwrap();
                black_box(results)
            })
        });
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_large_embeddings_memory,
    benchmark_large_embeddings_search
);
criterion_main!(benches);