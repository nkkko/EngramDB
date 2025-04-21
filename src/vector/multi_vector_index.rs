use std::collections::HashMap;
use uuid::Uuid;

use crate::core::MemoryNode;
use crate::embeddings::multi_vector::MultiVectorEmbedding;
use crate::error::EngramDbError;
use crate::Result;

use super::VectorSearchIndex;

/// A specialized vector index for multi-vector embeddings (ColBERT/ColPali-style)
#[derive(Clone)]
pub struct MultiVectorIndex {
    /// Map of memory node IDs to their multi-vector embeddings
    multi_vectors: HashMap<Uuid, MultiVectorEmbedding>,

    /// Configuration options for the index
    config: MultiVectorIndexConfig,
}

/// Configuration for the multi-vector index
#[derive(Debug, Clone)]
pub struct MultiVectorIndexConfig {
    /// Whether to quantize vectors for memory efficiency
    pub quantize: bool,

    /// The similarity scoring method to use
    pub similarity_method: MultiVectorSimilarityMethod,
}

/// Available similarity methods for multi-vector search
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MultiVectorSimilarityMethod {
    /// Maximum similarity between any pair of vectors (MaxSim)
    Maximum,

    /// Average similarity across all pairs of vectors
    Average,

    /// Late interaction similarity using ColBERT-style scoring
    LateInteraction,
}

impl Default for MultiVectorIndexConfig {
    fn default() -> Self {
        Self {
            quantize: false,
            similarity_method: MultiVectorSimilarityMethod::LateInteraction,
        }
    }
}

impl Default for MultiVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiVectorIndex {
    /// Create a new multi-vector index with default configuration
    pub fn new() -> Self {
        Self::with_config(MultiVectorIndexConfig::default())
    }

    /// Create a new multi-vector index with the given configuration
    pub fn with_config(config: MultiVectorIndexConfig) -> Self {
        Self {
            multi_vectors: HashMap::new(),
            config,
        }
    }

    /// Add a memory node to the index
    pub fn add(&mut self, node: &MemoryNode) -> Result<()> {
        // Check if this is a multi-vector node
        if let Some(multi_vec) = node.multi_vector_embeddings() {
            let mut multi_vec = multi_vec.clone();

            // Quantize if configured
            if self.config.quantize {
                multi_vec
                    .quantize()
                    .map_err(|e| EngramDbError::Vector(e.to_string()))?;
            }

            // Insert or replace the embeddings for this node
            self.multi_vectors.insert(node.id(), multi_vec);
            Ok(())
        } else {
            Err(EngramDbError::Vector(format!(
                "Memory node does not contain multi-vector embeddings: {}",
                node.id()
            )))
        }
    }

    /// Remove a memory node from the index
    pub fn remove(&mut self, id: Uuid) -> Result<()> {
        if self.multi_vectors.remove(&id).is_none() {
            return Err(EngramDbError::Vector(format!(
                "Memory node not found in multi-vector index: {}",
                id
            )));
        }
        Ok(())
    }

    /// Update a memory node in the index
    pub fn update(&mut self, node: &MemoryNode) -> Result<()> {
        // Reuse the same code path as add
        self.add(node)
    }

    /// Perform a search for similar vectors
    ///
    /// # Arguments
    ///
    /// * `query` - The query multi-vector
    /// * `limit` - Maximum number of results to return
    /// * `threshold` - Minimum similarity threshold (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A vector of (UUID, similarity) pairs, sorted by descending similarity
    pub fn search(
        &self,
        query: &MultiVectorEmbedding,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<(Uuid, f32)>> {
        let mut results = Vec::new();

        // Compute similarity for each multi-vector
        for (id, multi_vec) in &self.multi_vectors {
            let similarity = match self.config.similarity_method {
                MultiVectorSimilarityMethod::Maximum => query.max_similarity(multi_vec),
                MultiVectorSimilarityMethod::Average => query.avg_similarity(multi_vec),
                MultiVectorSimilarityMethod::LateInteraction => {
                    query.late_interaction_score(multi_vec)
                }
            };

            if similarity >= threshold {
                results.push((*id, similarity));
            }
        }

        // Sort by similarity (descending)
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Limit results
        if results.len() > limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Get the multi-vector embedding for a specific ID
    pub fn get(&self, id: Uuid) -> Option<&MultiVectorEmbedding> {
        self.multi_vectors.get(&id)
    }

    /// Get the number of multi-vectors in the index
    pub fn len(&self) -> usize {
        self.multi_vectors.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.multi_vectors.is_empty()
    }
}

// Implement VectorSearchIndex for MultiVectorIndex
impl VectorSearchIndex for MultiVectorIndex {
    fn add(&mut self, node: &MemoryNode) -> Result<()> {
        if node.is_multi_vector() {
            self.add(node)
        } else {
            // For single-vector nodes, return an error - they should use a different index
            Err(EngramDbError::Vector(format!(
                "Cannot add single-vector node to multi-vector index: {}",
                node.id()
            )))
        }
    }

    fn remove(&mut self, id: Uuid) -> Result<()> {
        self.remove(id)
    }

    fn update(&mut self, node: &MemoryNode) -> Result<()> {
        if node.is_multi_vector() {
            self.update(node)
        } else {
            Err(EngramDbError::Vector(format!(
                "Cannot update single-vector node in multi-vector index: {}",
                node.id()
            )))
        }
    }

    fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Result<Vec<(Uuid, f32)>> {
        // When called through the VectorSearchIndex trait, we need to convert the query to a multi-vector
        // This is a limited compatibility mode - for best results, use the specialized search method directly

        // Create a pseudo multi-vector with a single embedding
        let multi_query = MultiVectorEmbedding::from_single_vector(query.to_vec())
            .map_err(|e| EngramDbError::Vector(e.to_string()))?;

        self.search(&multi_query, limit, threshold)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn get(&self, id: Uuid) -> Option<&Vec<f32>> {
        // This is a compatibility method that doesn't have a perfect mapping
        // We need to return some vector for the standard interface, so return the first vector if available
        self.multi_vectors
            .get(&id)
            .and_then(|multi_vec| multi_vec.vectors().first().map(|v| v as &Vec<f32>))
    }
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::multi_vector::{MockMultiVectorProvider, MultiVectorProvider};

    #[test]
    fn test_multi_vector_index_crud() {
        let mut index = MultiVectorIndex::new();

        // Create test nodes with multi-vector embeddings
        let provider = MockMultiVectorProvider::new(96, 20);

        let mv1 = provider.generate_random_multi_vector(Some(20));
        let mv2 = provider.generate_random_multi_vector(Some(20));
        let mv3 = provider.generate_random_multi_vector(Some(20));

        let node1 = MemoryNode::with_multi_vector(mv1);
        let node2 = MemoryNode::with_multi_vector(mv2);
        let node3 = MemoryNode::with_multi_vector(mv3);

        // Add to index
        index.add(&node1).unwrap();
        index.add(&node2).unwrap();
        index.add(&node3).unwrap();

        assert_eq!(index.len(), 3);

        // Remove
        index.remove(node2.id()).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.get(node2.id()).is_none());

        // Update
        let mv1_updated = provider.generate_random_multi_vector(Some(20));
        let mut node1_updated = node1.clone();
        node1_updated.set_multi_vector_embeddings(mv1_updated.clone());

        index.update(&node1_updated).unwrap();

        let retrieved = index.get(node1.id()).unwrap();
        assert_eq!(retrieved, &mv1_updated);
    }

    #[test]
    fn test_multi_vector_search() {
        let mut index = MultiVectorIndex::new();
        let provider = MockMultiVectorProvider::new(96, 10);

        // Add some test nodes
        let num_nodes = 10;
        let mut nodes = Vec::with_capacity(num_nodes);

        for _ in 0..num_nodes {
            let multi_vec = provider.generate_random_multi_vector(None);
            let node = MemoryNode::with_multi_vector(multi_vec);
            index.add(&node).unwrap();
            nodes.push(node);
        }

        // Create a query multi-vector
        let query = provider.generate_random_multi_vector(Some(5));

        // Test search with different methods
        for method in &[
            MultiVectorSimilarityMethod::Maximum,
            MultiVectorSimilarityMethod::Average,
            MultiVectorSimilarityMethod::LateInteraction,
        ] {
            let mut config = MultiVectorIndexConfig::default();
            config.similarity_method = *method;

            let index_with_method = MultiVectorIndex::with_config(config);

            // Add all nodes to the new index
            let mut index_with_method = index_with_method;
            for node in &nodes {
                index_with_method.add(node).unwrap();
            }

            // Perform search
            let results = index_with_method.search(&query, 5, 0.0).unwrap();

            // Check that we got the expected number of results
            assert_eq!(results.len(), 5);

            // Check that all scores are between 0 and 1
            for (_, score) in &results {
                assert!(*score >= 0.0 && *score <= 1.0);
            }

            // Check that results are sorted in descending order
            for i in 1..results.len() {
                assert!(results[i - 1].1 >= results[i].1);
            }
        }
    }

    #[test]
    fn test_single_vector_compatibility() {
        let mut multi_index = MultiVectorIndex::new();

        // Create a node with single-vector embeddings
        let node = MemoryNode::new(vec![1.0, 0.0, 0.0]);

        // Adding should fail
        let result = multi_index.add(&node);
        assert!(result.is_err());
    }

    #[test]
    fn test_multi_vector_quantization() {
        let provider = MockMultiVectorProvider::new(96, 20);
        let multi_vec = provider.generate_random_multi_vector(None);
        let node = MemoryNode::with_multi_vector(multi_vec);

        // Create index with quantization enabled
        let mut config = MultiVectorIndexConfig::default();
        config.quantize = true;
        let mut index = MultiVectorIndex::with_config(config);

        // Add node to index (which will quantize it)
        index.add(&node).unwrap();

        // Check that the stored multi-vector is quantized
        let retrieved = index.get(node.id()).unwrap();
        assert!(retrieved.is_quantized());
    }
}
