use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rtamp::core::MemoryNode;
use rtamp::vector::VectorIndex;
use uuid::Uuid;

fn create_random_embedding(dim: usize) -> Vec<f32> {
    (0..dim).map(|_| rand::random::<f32>()).collect()
}

fn benchmark_vector_search(c: &mut Criterion) {
    // Create a vector index with varying numbers of embeddings
    let mut group = c.benchmark_group("vector_search");

    for &size in &[100, 1000, 10000] {
        // Create index with `size` random embeddings
        let mut index = VectorIndex::new();

        // Add random embeddings
        for _ in 0..size {
            let node = MemoryNode::new(create_random_embedding(128));
            index.add(&node).unwrap();
        }

        // Create a random query vector
        let query = create_random_embedding(128);

        group.bench_function(format!("search_{}", size), |b| {
            b.iter(|| {
                let results = index
                    .search(black_box(&query), black_box(10), black_box(0.0))
                    .unwrap();
                black_box(results)
            })
        });
    }

    group.finish();
}

fn benchmark_vector_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("vector_operations");

    // Benchmark adding vectors
    group.bench_function("add", |b| {
        let mut index = VectorIndex::new();

        b.iter(|| {
            let node = MemoryNode::new(create_random_embedding(128));
            index.add(black_box(&node)).unwrap();
            black_box(&index);
        })
    });

    // Benchmark updating vectors
    group.bench_function("update", |b| {
        let mut index = VectorIndex::new();
        let mut node = MemoryNode::new(create_random_embedding(128));
        index.add(&node).unwrap();

        b.iter(|| {
            node.set_embeddings(create_random_embedding(128));
            index.update(black_box(&node)).unwrap();
            black_box(&index);
        })
    });

    // Benchmark removing vectors
    group.bench_function("remove", |b| {
        b.iter_with_setup(
            || {
                // Setup: create an index with one node
                let mut index = VectorIndex::new();
                let node = MemoryNode::new(create_random_embedding(128));
                let id = node.id();
                index.add(&node).unwrap();
                (index, id)
            },
            |(mut index, id)| {
                index.remove(black_box(id)).unwrap();
                black_box(&index);
            },
        )
    });

    group.finish();
}

criterion_group!(
    benches,
    benchmark_vector_search,
    benchmark_vector_operations
);
criterion_main!(benches);
