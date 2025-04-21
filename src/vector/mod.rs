//! Vector search and similarity operations
//!
//! This module handles vector embedding operations including
//! similarity computations and retrieval by vector similarity.

mod similarity;
mod vector_index;
mod hnsw;
mod thread_safe;
mod multi_vector_index;

use crate::core::MemoryNode;
use crate::Result;
use uuid::Uuid;

pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use vector_index::VectorIndex;
pub use hnsw::{HnswIndex, HnswConfig};
pub use thread_safe::{ThreadSafeDatabase, ThreadSafeDatabasePool};
pub use multi_vector_index::{
    MultiVectorIndex, MultiVectorIndexConfig, MultiVectorSimilarityMethod
};

/// Available vector index algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VectorAlgorithm {
    /// Linear search (brute force)
    Linear,
    /// Hierarchical Navigable Small World
    HNSW,
    /// Multi-vector linear search (for ColBERT/ColPali-style)
    MultiVector,
}

/// Common trait for vector indices
pub trait VectorSearchIndex: Send + Sync {
    /// Adds a memory node to the index
    fn add(&mut self, node: &MemoryNode) -> Result<()>;
    
    /// Removes a memory node from the index
    fn remove(&mut self, id: Uuid) -> Result<()>;
    
    /// Updates a memory node in the index
    fn update(&mut self, node: &MemoryNode) -> Result<()>;
    
    /// Performs a similarity search to find the most similar vectors
    fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Result<Vec<(Uuid, f32)>>;
    
    /// Returns the number of vectors in the index
    fn len(&self) -> usize;
    
    /// Checks if the index is empty
    fn is_empty(&self) -> bool;
    
    /// Gets a reference to a vector by its ID
    fn get(&self, id: Uuid) -> Option<&Vec<f32>>;
}

/// Configuration for vector indices
#[derive(Debug, Clone)]
pub struct VectorIndexConfig {
    /// The algorithm to use
    pub algorithm: VectorAlgorithm,
    /// HNSW-specific configuration
    pub hnsw: Option<HnswConfig>,
    /// Multi-vector-specific configuration
    pub multi_vector: Option<MultiVectorIndexConfig>,
}

impl Default for VectorIndexConfig {
    fn default() -> Self {
        Self {
            algorithm: VectorAlgorithm::Linear,
            hnsw: None,
            multi_vector: None,
        }
    }
}

/// Create a new vector index based on the specified algorithm
pub fn create_vector_index(config: &VectorIndexConfig) -> Box<dyn VectorSearchIndex + Send + Sync> {
    match config.algorithm {
        VectorAlgorithm::Linear => Box::new(VectorIndex::new()),
        VectorAlgorithm::HNSW => {
            if let Some(hnsw_config) = config.hnsw {
                Box::new(HnswIndex::with_config(hnsw_config))
            } else {
                Box::new(HnswIndex::new())
            }
        },
        VectorAlgorithm::MultiVector => {
            if let Some(multi_vector_config) = &config.multi_vector {
                Box::new(MultiVectorIndex::with_config(multi_vector_config.clone()))
            } else {
                Box::new(MultiVectorIndex::new())
            }
        }
    }
}
