use crate::core::MemoryNode;
use crate::embeddings::multi_vector::MultiVectorEmbedding;
use crate::error::EngramDbError;
use crate::vector::{
    cosine_similarity, HnswConfig, MultiVectorIndexConfig, MultiVectorSimilarityMethod,
    VectorSearchIndex,
};
use crate::Result;

use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use uuid::Uuid;

/// Pooling strategy for combining multiple vectors into a single representation
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PoolingStrategy {
    /// Average pooling (mean of all vectors)
    MeanPool,

    /// Max pooling (element-wise maximum)
    MaxPool,

    /// Use centroid of the vectors
    Centroid,
}

impl Default for PoolingStrategy {
    fn default() -> Self {
        Self::MeanPool
    }
}

/// Configuration for the HNSW multi-vector index
#[derive(Debug, Clone)]
pub struct HnswMultiVectorConfig {
    /// Standard HNSW parameters
    pub hnsw_config: HnswConfig,

    /// Multi-vector specific parameters
    pub multi_vector_config: MultiVectorIndexConfig,

    /// Strategy for pooling vectors into a representative embedding
    pub pooling_strategy: PoolingStrategy,
}

impl Default for HnswMultiVectorConfig {
    fn default() -> Self {
        Self {
            hnsw_config: HnswConfig::default(),
            multi_vector_config: MultiVectorIndexConfig::default(),
            pooling_strategy: PoolingStrategy::default(),
        }
    }
}

/// A node in the HNSW graph for multi-vector index
#[derive(Debug, Clone)]
struct HnswNode {
    /// The UUID of the memory node
    id: Uuid,

    /// Representative embedding vector for HNSW routing
    representative: Vec<f32>,

    /// Connections per level (level -> [connected node IDs])
    connections: Vec<Vec<Uuid>>,
}

/// Represents a candidate node during search
#[derive(Debug, Clone, Copy)]
struct Candidate {
    /// ID of the candidate node
    id: Uuid,

    /// Distance/similarity to the query
    distance: f32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.distance == other.distance
    }
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // For max heap with BinaryHeap, we need to reverse the ordering
        // so that the closest (highest similarity) nodes are at the top
        other
            .distance
            .partial_cmp(&self.distance)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// HNSW adaptation optimized for multi-vector embeddings (ColBERT-style)
#[derive(Clone)]
pub struct HnswMultiVectorIndex {
    /// Map of node IDs to their multi-vector embeddings
    embeddings: HashMap<Uuid, MultiVectorEmbedding>,

    /// HNSW graph nodes with representative embeddings for routing
    nodes: HashMap<Uuid, HnswNode>,

    /// Configuration parameters
    config: HnswMultiVectorConfig,

    /// Entry point to the graph (node ID at the top level)
    entry_point: Option<Uuid>,

    /// Current maximum level in the graph
    max_level: usize,

    /// Random number generator for level generation
    rng: StdRng,
}

impl Default for HnswMultiVectorIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswMultiVectorIndex {
    /// Create a new HNSW multi-vector index with default configuration
    pub fn new() -> Self {
        Self::with_config(HnswMultiVectorConfig::default())
    }

    /// Create a new HNSW multi-vector index with the specified configuration
    pub fn with_config(config: HnswMultiVectorConfig) -> Self {
        Self {
            embeddings: HashMap::new(),
            nodes: HashMap::new(),
            config,
            entry_point: None,
            max_level: 0,
            rng: StdRng::from_entropy(),
        }
    }

    /// Add a multi-vector to the index
    pub fn add(&mut self, node: &MemoryNode) -> Result<()> {
        // Check if this is a multi-vector node
        if let Some(multi_vec) = node.multi_vector_embeddings() {
            self.add_multi_vector(node.id(), multi_vec.clone())
        } else {
            Err(EngramDbError::Vector(format!(
                "Memory node does not contain multi-vector embeddings: {}",
                node.id()
            )))
        }
    }

    /// Add a multi-vector embedding with the given ID
    pub fn add_multi_vector(
        &mut self,
        id: Uuid,
        mut multi_vec: MultiVectorEmbedding,
    ) -> Result<()> {
        // If the node already exists, update it
        if self.nodes.contains_key(&id) {
            return self.update_multi_vector(id, multi_vec);
        }

        // Quantize if configured
        if self.config.multi_vector_config.quantize {
            multi_vec
                .quantize()
                .map_err(|e| EngramDbError::Vector(e.to_string()))?;
        }

        // Store the multi-vector embedding
        self.embeddings.insert(id, multi_vec.clone());

        // Create a representative vector for HNSW routing
        let representative = self.create_representative_vector(&multi_vec);

        // Generate a random level for the new node
        let level = self.get_random_level();

        // Create the new node with empty connections
        let mut connections = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            // Pre-allocate some capacity for connections
            connections.push(Vec::with_capacity(self.config.hnsw_config.m));
        }

        // Store a copy of the representative for computing similarity later
        let representative_copy = representative.clone();

        let new_node = HnswNode {
            id,
            representative,
            connections,
        };

        // If this is the first node, make it the entry point
        if self.nodes.is_empty() {
            self.nodes.insert(id, new_node);
            self.entry_point = Some(id);
            self.max_level = level;
            return Ok(());
        }

        // Get the entry point
        let entry_point_id = self.entry_point.expect("Entry point should exist");

        // Insert the new node
        self.nodes.insert(id, new_node);

        // Connect the new node to the graph
        let mut curr_node_id = entry_point_id;
        let similarity = self.compute_similarity(&representative_copy, entry_point_id)?;
        let mut curr_dist = similarity.expect("Entry point should exist");

        // For each level, find the closest node and add connections
        for level_idx in (0..=level.min(self.max_level)).rev() {
            // Search for the closest neighbors at the current level
            let nearest =
                self.search_level(&representative_copy, curr_node_id, curr_dist, level_idx, 1)?;

            if !nearest.is_empty() {
                curr_node_id = nearest[0].id;
                curr_dist = nearest[0].distance;
            }

            // If we're at a level where the new node exists, connect it
            if level_idx <= level {
                // Get ef_construction nearest neighbors at this level
                let neighbors = self.search_level(
                    &representative_copy,
                    curr_node_id,
                    curr_dist,
                    level_idx,
                    self.config.hnsw_config.ef_construction,
                )?;

                // Select the best M neighbors and connect to them
                let selected = self.select_neighbors(&neighbors, self.config.hnsw_config.m);

                // Add bidirectional connections
                for &neighbor_id in &selected {
                    // Connect new node to neighbor
                    if let Some(node) = self.nodes.get_mut(&id) {
                        if !node.connections[level_idx].contains(&neighbor_id) {
                            node.connections[level_idx].push(neighbor_id);
                        }
                    }

                    // Connect neighbor to new node
                    if let Some(node) = self.nodes.get_mut(&neighbor_id) {
                        if node.connections.len() > level_idx {
                            if !node.connections[level_idx].contains(&id) {
                                node.connections[level_idx].push(id);
                            }

                            // Ensure neighbor doesn't have too many connections
                            if node.connections[level_idx].len() > self.config.hnsw_config.m {
                                // We need to get all the connection IDs before we drop the mutable borrow
                                let conn_ids: Vec<Uuid> = node.connections[level_idx].clone();
                                // Clone the embedding for later use
                                let neighbor_representative = node.representative.clone();
                                // End the mutable borrow scope
                                let _ = node;

                                // Pre-allocate candidates
                                let mut candidates = Vec::with_capacity(conn_ids.len());
                                // Calculate similarities outside of mutable borrow
                                for conn_id in conn_ids {
                                    if let Ok(Some(sim)) =
                                        self.compute_similarity(&neighbor_representative, conn_id)
                                    {
                                        candidates.push(Candidate {
                                            id: conn_id,
                                            distance: sim,
                                        });
                                    }
                                }

                                let selected =
                                    self.select_neighbors(&candidates, self.config.hnsw_config.m);

                                // Update connections
                                if let Some(node) = self.nodes.get_mut(&neighbor_id) {
                                    node.connections[level_idx] = selected;
                                }
                            }
                        }
                    }
                }
            }
        }

        // Update the max level and entry point if necessary
        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id);
        }

        Ok(())
    }

    /// Update a multi-vector in the index
    pub fn update_multi_vector(&mut self, id: Uuid, multi_vec: MultiVectorEmbedding) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(EngramDbError::Vector(format!(
                "Node not found in index: {}",
                id
            )));
        }

        // For HNSW, we'll remove and re-add with the new multi-vector
        self.remove(id)?;
        self.add_multi_vector(id, multi_vec)?;

        Ok(())
    }

    /// Remove a node from the index
    pub fn remove(&mut self, id: Uuid) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(EngramDbError::Vector(format!(
                "Node not found in index: {}",
                id
            )));
        }

        // Get the node's connections before removing it
        let mut level_connections = Vec::new();
        if let Some(node) = self.nodes.get(&id) {
            // Pre-allocate with the exact number of levels
            level_connections.reserve(node.connections.len());

            for level in 0..node.connections.len() {
                // For each level, only collect the IDs we need to process
                let connected_ids = node.connections[level].to_vec();
                level_connections.push(connected_ids);
            }
        } else {
            return Ok(());
        }

        // Remove the node
        self.nodes.remove(&id);
        self.embeddings.remove(&id);

        // Update connections for all connected nodes
        for (level, connected_ids) in level_connections.into_iter().enumerate() {
            for connected_id in connected_ids {
                if let Some(node) = self.nodes.get_mut(&connected_id) {
                    if node.connections.len() > level {
                        node.connections[level].retain(|&conn_id| conn_id != id);
                    }
                }
            }
        }

        // If the entry point was removed, find a new one
        if self.entry_point == Some(id) {
            self.entry_point = None;

            // Find the node with the highest level
            let mut max_level = 0;
            for (node_id, node) in &self.nodes {
                if node.connections.len() > max_level {
                    max_level = node.connections.len();
                    self.entry_point = Some(*node_id);
                }
            }

            self.max_level = max_level.saturating_sub(1);
        }

        Ok(())
    }

    /// Search for similar multi-vectors using two-phase approach
    pub fn search_multi_vector(
        &self,
        query: &MultiVectorEmbedding,
        limit: usize,
        threshold: f32,
    ) -> Result<Vec<(Uuid, f32)>> {
        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Phase 1: Create a representative vector for routing
        let representative = self.create_representative_vector(query);
        
        // Create a copy for search operations
        let representative_copy = representative.clone();

        // Get the entry point
        let entry_point_id = self.entry_point.expect("Entry point should exist");
        let similarity = self.compute_similarity(&representative_copy, entry_point_id)?;
        let entry_dist = similarity.expect("Entry point should exist");

        // Start from the top level and traverse down
        let mut curr_node_id = entry_point_id;
        let mut curr_dist = entry_dist;

        // For each level, find the closest node
        for level_idx in (1..=self.max_level).rev() {
            let nearest =
                self.search_level(&representative_copy, curr_node_id, curr_dist, level_idx, 1)?;
            if !nearest.is_empty() {
                curr_node_id = nearest[0].id;
                curr_dist = nearest[0].distance;
            }
        }

        // Perform the main search at level 0
        let ef = limit.max(self.config.hnsw_config.ef);
        let candidates = self.search_level(&representative_copy, curr_node_id, curr_dist, 0, ef)?;

        // Phase 2: Re-rank candidates using full multi-vector similarity
        let mut results = Vec::with_capacity(candidates.len());

        for candidate in candidates {
            if let Some(multi_vec) = self.embeddings.get(&candidate.id) {
                let similarity = match self.config.multi_vector_config.similarity_method {
                    MultiVectorSimilarityMethod::Maximum => query.max_similarity(multi_vec),
                    MultiVectorSimilarityMethod::Average => query.avg_similarity(multi_vec),
                    MultiVectorSimilarityMethod::LateInteraction => {
                        query.late_interaction_score(multi_vec)
                    }
                };

                if similarity >= threshold {
                    results.push((candidate.id, similarity));
                }
            }
        }

        // Sort by similarity
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));

        // Limit results
        if results.len() > limit {
            results.truncate(limit);
        }

        Ok(results)
    }

    /// Create a representative vector from a multi-vector embedding
    fn create_representative_vector(&self, multi_vec: &MultiVectorEmbedding) -> Vec<f32> {
        match self.config.pooling_strategy {
            PoolingStrategy::MeanPool => {
                // Mean pooling: average all vectors
                let vectors = multi_vec.vectors();
                let dim = multi_vec.dimensions();
                let mut result = vec![0.0; dim];

                for vec in vectors {
                    for i in 0..dim {
                        result[i] += vec[i];
                    }
                }

                // Normalize by the number of vectors
                let num_vectors = vectors.len() as f32;
                for val in &mut result {
                    *val /= num_vectors;
                }

                // Ensure the vector is normalized
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut result {
                        *val /= norm;
                    }
                }

                result
            }
            PoolingStrategy::MaxPool => {
                // Max pooling: element-wise maximum
                let vectors = multi_vec.vectors();
                if vectors.is_empty() {
                    return Vec::new();
                }

                let dim = multi_vec.dimensions();
                let mut result = vec![std::f32::NEG_INFINITY; dim];

                for vec in vectors {
                    for i in 0..dim {
                        result[i] = result[i].max(vec[i]);
                    }
                }

                // Ensure the vector is normalized
                let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 0.0 {
                    for val in &mut result {
                        *val /= norm;
                    }
                }

                result
            }
            PoolingStrategy::Centroid => {
                // Find the centroid (vector closest to all others)
                let vectors = multi_vec.vectors();
                if vectors.is_empty() {
                    return Vec::new();
                } else if vectors.len() == 1 {
                    return vectors[0].clone();
                }

                // Find the vector with highest average cosine similarity to others
                let mut best_idx = 0;
                let mut best_avg_sim = 0.0;

                for i in 0..vectors.len() {
                    let mut total_sim = 0.0;
                    let mut count = 0;

                    for j in 0..vectors.len() {
                        if i != j {
                            if let Some(sim) = cosine_similarity(&vectors[i], &vectors[j]) {
                                total_sim += sim;
                                count += 1;
                            }
                        }
                    }

                    let avg_sim = if count > 0 {
                        total_sim / count as f32
                    } else {
                        0.0
                    };
                    if avg_sim > best_avg_sim {
                        best_avg_sim = avg_sim;
                        best_idx = i;
                    }
                }

                vectors[best_idx].clone()
            }
        }
    }

    /// Number of nodes in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a multi-vector embedding by its ID
    pub fn get(&self, id: Uuid) -> Option<&MultiVectorEmbedding> {
        self.embeddings.get(&id)
    }

    /// Generate a random level according to the HNSW distribution
    fn get_random_level(&mut self) -> usize {
        let r: f32 = self.rng.gen();
        (-r.ln() * self.config.hnsw_config.level_multiplier) as usize
    }

    /// Compute similarity between a query vector and a node's representative
    fn compute_similarity(&self, query: &[f32], node_id: Uuid) -> Result<Option<f32>> {
        if let Some(node) = self.nodes.get(&node_id) {
            Ok(cosine_similarity(query, &node.representative))
        } else {
            Err(EngramDbError::Vector(format!(
                "Node not found in index: {}",
                node_id
            )))
        }
    }

    /// Search for nearest neighbors at a specific level
    fn search_level(
        &self,
        query: &[f32],
        entry_id: Uuid,
        entry_dist: f32,
        level: usize,
        ef: usize,
    ) -> Result<Vec<Candidate>> {
        // Initialize visited set
        let mut visited = HashSet::with_capacity(ef * 2);
        visited.insert(entry_id);

        // Initialize candidate heap (max heap for candidates)
        let mut candidates = std::collections::BinaryHeap::with_capacity(ef * 2);
        candidates.push(Candidate {
            id: entry_id,
            distance: entry_dist,
        });

        // Initialize results heap (min heap for results)
        let mut results = std::collections::BinaryHeap::with_capacity(ef + 1);
        results.push(Candidate {
            id: entry_id,
            distance: -entry_dist, // Negative to make it a min heap
        });

        // Search until candidates is empty or the best candidate is worse than the worst result
        while !candidates.is_empty() {
            // Get the best candidate
            let curr = candidates.pop().unwrap();

            // If we have enough results and the current candidate is worse than our worst result,
            // we can stop searching
            if !results.is_empty() && -results.peek().unwrap().distance > curr.distance {
                break;
            }

            // Get the current node's connections at this level
            let connected_ids = if let Some(node) = self.nodes.get(&curr.id) {
                if node.connections.len() > level {
                    node.connections[level].clone()
                } else {
                    Vec::new()
                }
            } else {
                Vec::new()
            };

            // Explore the neighbors
            for neighbor_id in connected_ids {
                if visited.insert(neighbor_id) {
                    // Compute similarity with the query
                    if let Some(sim) = self.compute_similarity(query, neighbor_id)? {
                        // Get the current worst result
                        let worst_result_dist = if results.len() >= ef {
                            Some(-results.peek().unwrap().distance)
                        } else {
                            None
                        };

                        // If we have fewer than ef results or this neighbor is better than the worst result,
                        // add it to the candidates and results
                        if worst_result_dist.is_none() || sim > worst_result_dist.unwrap() {
                            candidates.push(Candidate {
                                id: neighbor_id,
                                distance: sim,
                            });

                            results.push(Candidate {
                                id: neighbor_id,
                                distance: -sim, // Negative for min heap
                            });

                            // If we have too many results, remove the worst one
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        // Convert results to a vector, reversing the negative distance
        let mut result_vec = Vec::with_capacity(results.len());
        while let Some(res) = results.pop() {
            result_vec.push(Candidate {
                id: res.id,
                distance: -res.distance, // Convert back to positive similarity
            });
        }

        // Sort by descending similarity
        result_vec.sort_by(|a, b| {
            b.distance
                .partial_cmp(&a.distance)
                .unwrap_or(Ordering::Equal)
        });

        Ok(result_vec)
    }

    /// Select the best M neighbors from candidates
    fn select_neighbors(&self, candidates: &[Candidate], m: usize) -> Vec<Uuid> {
        // If we have fewer candidates than M, use all of them
        if candidates.len() <= m {
            candidates.iter().map(|c| c.id).collect()
        } else {
            // Take the top M candidates by similarity
            candidates.iter().take(m).map(|c| c.id).collect()
        }
    }
}

impl VectorSearchIndex for HnswMultiVectorIndex {
    fn add(&mut self, node: &MemoryNode) -> Result<()> {
        if node.is_multi_vector() {
            self.add(node)
        } else {
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
            if let Some(multi_vec) = node.multi_vector_embeddings() {
                self.update_multi_vector(node.id(), multi_vec.clone())
            } else {
                Err(EngramDbError::Vector(format!(
                    "Memory node does not contain multi-vector embeddings: {}",
                    node.id()
                )))
            }
        } else {
            Err(EngramDbError::Vector(format!(
                "Cannot update single-vector node in multi-vector index: {}",
                node.id()
            )))
        }
    }

    fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Result<Vec<(Uuid, f32)>> {
        // Create a pseudo multi-vector with a single vector for compatibility
        let multi_query = MultiVectorEmbedding::from_single_vector(query.to_vec())
            .map_err(|e| EngramDbError::Vector(e.to_string()))?;

        self.search_multi_vector(&multi_query, limit, threshold)
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn is_empty(&self) -> bool {
        self.is_empty()
    }

    fn get(&self, id: Uuid) -> Option<&Vec<f32>> {
        // This is a compatibility method for the interface
        // We have to return some vector, so return the representative if available
        self.nodes.get(&id).map(|node| &node.representative)
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
    fn test_hnsw_multi_vector_basic() {
        let mut index = HnswMultiVectorIndex::new();

        // Create a multi-vector provider for test data
        let provider = MockMultiVectorProvider::new(96, 20);

        // Create test nodes with multi-vector embeddings
        let mv1 = provider.generate_random_multi_vector(Some(20));
        let mv2 = provider.generate_random_multi_vector(Some(20));
        let mv3 = provider.generate_random_multi_vector(Some(20));

        let node1 = MemoryNode::with_multi_vector(mv1.clone());
        let node2 = MemoryNode::with_multi_vector(mv2.clone());
        let node3 = MemoryNode::with_multi_vector(mv3.clone());

        // Add to index
        index.add(&node1).unwrap();
        index.add(&node2).unwrap();
        index.add(&node3).unwrap();

        assert_eq!(index.len(), 3);

        // Search
        let query = provider.generate_random_multi_vector(Some(10));
        let results = index.search_multi_vector(&query, 2, 0.0).unwrap();

        assert_eq!(results.len(), 2);
        assert!(results[0].1 >= 0.0 && results[0].1 <= 1.0);
        assert!(results[1].1 >= 0.0 && results[1].1 <= 1.0);

        // Check that scores are in descending order
        assert!(results[0].1 >= results[1].1);

        // Remove
        index.remove(node2.id()).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.get(node2.id()).is_none());

        // Update
        let mv1_updated = provider.generate_random_multi_vector(Some(20));
        let mut node1_updated = node1.clone();
        node1_updated.set_multi_vector_embeddings(mv1_updated.clone());

        index.update(&node1_updated).unwrap();
        assert_eq!(index.len(), 2);

        let retrieved = index.get(node1.id()).unwrap();
        assert_eq!(retrieved, &mv1_updated);
    }

    #[test]
    fn test_representative_vector_creation() {
        // Create a multi-vector provider for test data
        let provider = MockMultiVectorProvider::new(96, 20);

        // Create a multi-vector for testing
        let multi_vec = provider.generate_random_multi_vector(Some(5));

        // Test each pooling strategy
        let mut config = HnswMultiVectorConfig::default();

        // Mean pooling
        config.pooling_strategy = PoolingStrategy::MeanPool;
        let index = HnswMultiVectorIndex::with_config(config.clone());
        let mean_rep = index.create_representative_vector(&multi_vec);
        assert_eq!(mean_rep.len(), 96);

        // Max pooling
        config.pooling_strategy = PoolingStrategy::MaxPool;
        let index = HnswMultiVectorIndex::with_config(config.clone());
        let max_rep = index.create_representative_vector(&multi_vec);
        assert_eq!(max_rep.len(), 96);

        // Centroid
        config.pooling_strategy = PoolingStrategy::Centroid;
        let index = HnswMultiVectorIndex::with_config(config);
        let centroid_rep = index.create_representative_vector(&multi_vec);
        assert_eq!(centroid_rep.len(), 96);

        // Ensure all representations are different
        assert_ne!(mean_rep, max_rep);
        assert_ne!(mean_rep, centroid_rep);
        assert_ne!(max_rep, centroid_rep);
    }

    #[test]
    fn test_pooling_strategies() {
        // Create test vectors for a multi-vector
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let multi_vec = MultiVectorEmbedding::new(vectors).unwrap();

        // Test mean pooling
        let mut config = HnswMultiVectorConfig::default();
        config.pooling_strategy = PoolingStrategy::MeanPool;
        let index = HnswMultiVectorIndex::with_config(config);

        let mean_rep = index.create_representative_vector(&multi_vec);

        // The mean should be close to [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
        let expected_val = 1.0 / 3.0_f32.sqrt();
        assert!((mean_rep[0] - expected_val).abs() < 1e-5);
        assert!((mean_rep[1] - expected_val).abs() < 1e-5);
        assert!((mean_rep[2] - expected_val).abs() < 1e-5);

        // Test max pooling
        let mut config = HnswMultiVectorConfig::default();
        config.pooling_strategy = PoolingStrategy::MaxPool;
        let index = HnswMultiVectorIndex::with_config(config);

        let max_rep = index.create_representative_vector(&multi_vec);

        // The max should be close to [1/sqrt(3), 1/sqrt(3), 1/sqrt(3)]
        // Since we normalize after taking the max of each dimension
        let expected_val = 1.0 / 3.0_f32.sqrt();
        assert!((max_rep[0] - expected_val).abs() < 1e-5);
        assert!((max_rep[1] - expected_val).abs() < 1e-5);
        assert!((max_rep[2] - expected_val).abs() < 1e-5);

        // Test centroid
        let mut config = HnswMultiVectorConfig::default();
        config.pooling_strategy = PoolingStrategy::Centroid;
        let index = HnswMultiVectorIndex::with_config(config);

        let centroid_rep = index.create_representative_vector(&multi_vec);

        // The centroid should be one of the original vectors
        let is_original =
            (centroid_rep[0] == 1.0 && centroid_rep[1] == 0.0 && centroid_rep[2] == 0.0)
                || (centroid_rep[0] == 0.0 && centroid_rep[1] == 1.0 && centroid_rep[2] == 0.0)
                || (centroid_rep[0] == 0.0 && centroid_rep[1] == 0.0 && centroid_rep[2] == 1.0);

        assert!(is_original);
    }

    #[test]
    fn test_search_multi_vector() {
        let mut index = HnswMultiVectorIndex::new();

        // Create a multi-vector provider for test data
        let provider = MockMultiVectorProvider::new(96, 20);

        // Create test nodes with multi-vector embeddings
        let count = 30;
        let mut nodes = Vec::with_capacity(count);

        for _ in 0..count {
            let multi_vec = provider.generate_random_multi_vector(None);
            let node = MemoryNode::with_multi_vector(multi_vec);
            index.add(&node).unwrap();
            nodes.push(node);
        }

        // Create a query
        let query = provider.generate_random_multi_vector(Some(10));

        // Test with different pooling strategies
        for pooling in &[
            PoolingStrategy::MeanPool,
            PoolingStrategy::MaxPool,
            PoolingStrategy::Centroid,
        ] {
            let mut config = HnswMultiVectorConfig::default();
            config.pooling_strategy = *pooling;

            let mut index_with_strategy = HnswMultiVectorIndex::with_config(config);

            // Add all nodes to the index
            for node in &nodes {
                index_with_strategy.add(node).unwrap();
            }

            // Search
            let results = index_with_strategy
                .search_multi_vector(&query, 5, 0.0)
                .unwrap();

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
    fn test_hnsw_multi_vector_trait() {
        let mut index: Box<dyn VectorSearchIndex> = Box::new(HnswMultiVectorIndex::new());

        // Create a multi-vector provider for test data
        let provider = MockMultiVectorProvider::new(96, 20);

        // Create test node with multi-vector embeddings
        let multi_vec = provider.generate_random_multi_vector(None);
        let node = MemoryNode::with_multi_vector(multi_vec);

        // Add to index
        index.add(&node).unwrap();
        assert_eq!(index.len(), 1);

        // Check that the compatibility methods work
        assert!(index.get(node.id()).is_some());
    }
}
