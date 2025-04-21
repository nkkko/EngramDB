use std::collections::HashMap;
use uuid::Uuid;

use super::{cosine_similarity, VectorSearchIndex};
use crate::core::MemoryNode;
use crate::error::EngramDbError;
use crate::Result;

/// A simple vector index for similarity search
///
/// This is a basic implementation for the MVP that performs linear search
/// through the vectors. More advanced indexing methods would be implemented
/// in future versions.
#[derive(Default, Clone)]
pub struct VectorIndex {
    /// Map of memory node IDs to their embedding vectors
    embeddings: HashMap<Uuid, Vec<f32>>,
}

impl VectorIndex {
    /// Creates a new empty vector index
    pub fn new() -> Self {
        Default::default()
    }

    /// Adds a memory node to the index
    ///
    /// # Arguments
    ///
    /// * `node` - The memory node to add
    ///
    /// # Returns
    ///
    /// A Result indicating success or the specific error that occurred
    pub fn add(&mut self, node: &MemoryNode) -> Result<()> {
        // Get embeddings from the node
        if let Some(embeddings) = node.embeddings() {
            // Create a capacity-optimized vector to avoid over-allocation
            let mut vec = Vec::with_capacity(embeddings.len());
            vec.extend_from_slice(embeddings);

            // Insert or replace the embeddings for this node
            self.embeddings.insert(node.id(), vec);
            Ok(())
        } else {
            // This is a multi-vector node, which we can't handle
            Err(EngramDbError::Vector(format!(
                "Cannot add multi-vector embeddings to single-vector index: {}",
                node.id()
            )))
        }
    }

    /// Removes a memory node from the index
    ///
    /// # Arguments
    ///
    /// * `id` - The UUID of the memory node to remove
    ///
    /// # Returns
    ///
    /// A Result indicating success or the specific error that occurred
    pub fn remove(&mut self, id: Uuid) -> Result<()> {
        if self.embeddings.remove(&id).is_none() {
            return Err(EngramDbError::Vector(format!(
                "Memory node not found in index: {}",
                id
            )));
        }
        Ok(())
    }

    /// Updates a memory node in the index
    ///
    /// # Arguments
    ///
    /// * `node` - The memory node to update
    ///
    /// # Returns
    ///
    /// A Result indicating success or the specific error that occurred
    pub fn update(&mut self, node: &MemoryNode) -> Result<()> {
        // Reuse the same code path as add, which is optimized
        self.add(node)
    }

    /// Performs a similarity search to find the most similar vectors
    ///
    /// # Arguments
    ///
    /// * `query` - The query vector
    /// * `limit` - Maximum number of results to return
    /// * `threshold` - Minimum similarity threshold (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A vector of (UUID, similarity) pairs, sorted by descending similarity
    pub fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Result<Vec<(Uuid, f32)>> {
        if query.is_empty() {
            return Err(EngramDbError::Vector(
                "Query vector cannot be empty".to_string(),
            ));
        }

        let mut results = Vec::new();

        // Linear search through all vectors (naive approach for MVP)
        for (id, embedding) in &self.embeddings {
            if let Some(similarity) = cosine_similarity(query, embedding) {
                if similarity >= threshold {
                    results.push((*id, similarity));
                }
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

    /// Returns the number of vectors in the index
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    /// Checks if the index is empty
    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Gets a reference to a vector by its ID
    ///
    /// # Arguments
    ///
    /// * `id` - The UUID of the memory node
    ///
    /// # Returns
    ///
    /// A reference to the vector if found, otherwise None
    pub fn get(&self, id: Uuid) -> Option<&Vec<f32>> {
        self.embeddings.get(&id)
    }
}

// Implement the VectorSearchIndex trait for VectorIndex
impl VectorSearchIndex for VectorIndex {
    fn add(&mut self, node: &MemoryNode) -> Result<()> {
        self.add(node)
    }

    fn remove(&mut self, id: Uuid) -> Result<()> {
        self.remove(id)
    }

    fn update(&mut self, node: &MemoryNode) -> Result<()> {
        self.update(node)
    }

    fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Result<Vec<(Uuid, f32)>> {
        self.search(query, limit, threshold)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn get(&self, id: Uuid) -> Option<&Vec<f32>> {
        self.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::MemoryNode;

    #[test]
    fn test_vector_index_crud() {
        let mut index = VectorIndex::new();

        // Create test nodes
        let node1 = MemoryNode::new(vec![1.0, 0.0, 0.0]);
        let node2 = MemoryNode::new(vec![0.0, 1.0, 0.0]);
        let node3 = MemoryNode::new(vec![0.0, 0.0, 1.0]);

        // Add to index
        index.add(&node1).unwrap();
        index.add(&node2).unwrap();
        index.add(&node3).unwrap();

        assert_eq!(index.len(), 3);

        // Retrieve
        let vec1 = index.get(node1.id()).unwrap();
        assert_eq!(vec1, &vec![1.0, 0.0, 0.0]);

        // Remove
        index.remove(node2.id()).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.get(node2.id()).is_none());

        // Update
        let mut node1_updated = node1.clone();
        node1_updated.set_embeddings(vec![0.5, 0.5, 0.0]);
        index.update(&node1_updated).unwrap();

        let vec1_updated = index.get(node1.id()).unwrap();
        assert_eq!(vec1_updated, &vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn test_vector_search() {
        let mut index = VectorIndex::new();

        // Create test nodes with different but related vectors
        let node1 = MemoryNode::new(vec![1.0, 0.0, 0.0]); // X-axis
        let node2 = MemoryNode::new(vec![0.7, 0.7, 0.0]); // 45 degrees in XY plane
        let node3 = MemoryNode::new(vec![0.0, 1.0, 0.0]); // Y-axis
        let node4 = MemoryNode::new(vec![0.0, 0.0, 1.0]); // Z-axis

        // Add to index
        index.add(&node1).unwrap();
        index.add(&node2).unwrap();
        index.add(&node3).unwrap();
        index.add(&node4).unwrap();

        // Search for something closer to the X-axis
        let query = vec![0.9, 0.1, 0.0];
        let results = index.search(&query, 2, 0.0).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, node1.id()); // X-axis should be closest
        assert_eq!(results[1].0, node2.id()); // 45 degrees should be second

        // Search with threshold
        let results = index.search(&query, 10, 0.9).unwrap();
        assert_eq!(results.len(), 1); // Only node1 should be above 0.9 similarity

        // Search with empty query
        let query = vec![];
        let result = index.search(&query, 10, 0.0);
        assert!(result.is_err());
    }
}
