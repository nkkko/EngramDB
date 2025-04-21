use crate::core::MemoryNode;
use crate::error::EngramDbError;
use crate::vector::{cosine_similarity, VectorSearchIndex};
use crate::Result;
use rand::prelude::*;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use uuid::Uuid;

/// A node in the HNSW graph
#[derive(Debug, Clone)]
struct HnswNode {
    /// The UUID of the memory node
    #[allow(dead_code)]
    id: Uuid,
    /// The embedding vector (capacity-optimized to prevent memory waste)
    embedding: Vec<f32>,
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

/// Configuration for the HNSW algorithm
#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Maximum number of connections per node per layer
    pub m: usize,
    /// Size of the dynamic candidate list during construction
    pub ef_construction: usize,
    /// Size of the dynamic candidate list during search
    pub ef: usize,
    /// Base level for the multi-layer construction
    pub level_multiplier: f32,
    /// Maximum level in the hierarchy
    pub max_level: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        Self {
            m: 16,
            ef_construction: 100,
            ef: 10,
            level_multiplier: 1.0 / std::f32::consts::LN_2,
            max_level: 16,
        }
    }
}

/// Hierarchical Navigable Small World (HNSW) graph for fast vector search
///
/// This is an implementation of the HNSW algorithm described in:
/// "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs"
/// by Yu. A. Malkov, D. A. Yashunin (2016)
#[derive(Clone)]
pub struct HnswIndex {
    /// Map of node IDs to HNSW nodes
    nodes: HashMap<Uuid, HnswNode>,
    /// Configuration parameters
    config: HnswConfig,
    /// Entry point to the graph (node ID at the top level)
    entry_point: Option<Uuid>,
    /// Current maximum level in the graph
    max_level: usize,
    /// Random number generator for level generation
    rng: StdRng,
}

impl Default for HnswIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl HnswIndex {
    /// Create a new HNSW index with default configuration
    pub fn new() -> Self {
        Self::with_config(HnswConfig::default())
    }

    /// Create a new HNSW index with the specified configuration
    pub fn with_config(config: HnswConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            config,
            entry_point: None,
            max_level: 0,
            rng: StdRng::from_entropy(),
        }
    }

    /// Add a vector to the index
    pub fn add(&mut self, id: Uuid, embedding: Vec<f32>) -> Result<()> {
        // If the node already exists, update it
        if self.nodes.contains_key(&id) {
            return self.update(id, embedding);
        }

        // Generate a random level for the new node
        let level = self.get_random_level();

        // Create the new node with empty connections
        let mut connections = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            // Pre-allocate some capacity for connections to avoid frequent reallocations
            connections.push(Vec::with_capacity(self.config.m));
        }

        // Create a capacity-optimized embedding vector
        let mut optimized_embedding = Vec::with_capacity(embedding.len());
        optimized_embedding.extend_from_slice(&embedding);

        let new_node = HnswNode {
            id,
            embedding: optimized_embedding,
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
        let similarity = self.compute_similarity(&embedding, entry_point_id)?;
        let mut curr_dist = similarity.expect("Entry point should exist");

        // For each level, find the closest node and add connections
        for level_idx in (0..=level.min(self.max_level)).rev() {
            // Search for the closest neighbors at the current level
            let nearest = self.search_level(&embedding, curr_node_id, curr_dist, level_idx, 1)?;

            if !nearest.is_empty() {
                curr_node_id = nearest[0].id;
                curr_dist = nearest[0].distance;
            }

            // If we're at a level where the new node exists, connect it
            if level_idx <= level {
                // Get ef_construction nearest neighbors at this level
                let neighbors = self.search_level(
                    &embedding,
                    curr_node_id,
                    curr_dist,
                    level_idx,
                    self.config.ef_construction,
                )?;

                // Select the best M neighbors and connect to them
                let selected = self.select_neighbors(&neighbors, self.config.m);

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
                            if node.connections[level_idx].len() > self.config.m {
                                // We need to get all the connection IDs before we drop the mutable borrow
                                let conn_ids: Vec<Uuid> = node.connections[level_idx].clone();
                                // Clone the embedding for later use
                                let neighbor_embedding = node.embedding.clone();
                                // End the mutable borrow scope
                                let _ = node;

                                // Pre-allocate candidates
                                let mut candidates = Vec::with_capacity(conn_ids.len());
                                // Calculate similarities outside of mutable borrow
                                for conn_id in conn_ids {
                                    if let Ok(Some(sim)) =
                                        self.compute_similarity(&neighbor_embedding, conn_id)
                                    {
                                        candidates.push(Candidate {
                                            id: conn_id,
                                            distance: sim,
                                        });
                                    }
                                }

                                let selected = self.select_neighbors(&candidates, self.config.m);

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

    /// Update a vector in the index
    pub fn update(&mut self, id: Uuid, embedding: Vec<f32>) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(EngramDbError::Vector(format!(
                "Node not found in index: {}",
                id
            )));
        }

        // For HNSW, we'll remove and re-add with the optimized vector
        self.remove(id)?;

        // Create a capacity-optimized embedding vector
        let mut optimized_embedding = Vec::with_capacity(embedding.len());
        optimized_embedding.extend_from_slice(&embedding);

        self.add(id, optimized_embedding)?;

        Ok(())
    }

    /// Remove a vector from the index
    pub fn remove(&mut self, id: Uuid) -> Result<()> {
        if !self.nodes.contains_key(&id) {
            return Err(EngramDbError::Vector(format!(
                "Node not found in index: {}",
                id
            )));
        }

        // Get the node's connections before removing it without cloning the entire structure
        // Instead, collect the necessary information in a memory-efficient way
        let mut level_connections = Vec::new();
        if let Some(node) = self.nodes.get(&id) {
            // Pre-allocate with the exact number of levels
            level_connections.reserve(node.connections.len());

            for level in 0..node.connections.len() {
                // For each level, only collect the IDs we need to process
                // This avoids cloning large data structures
                let connected_ids = node.connections[level].to_vec();
                level_connections.push(connected_ids);
            }
        } else {
            return Ok(());
        }

        // Remove the node
        self.nodes.remove(&id);

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

    /// Search for similar vectors
    pub fn search(&self, query: &[f32], limit: usize, threshold: f32) -> Result<Vec<(Uuid, f32)>> {
        if query.is_empty() {
            return Err(EngramDbError::Vector(
                "Query vector cannot be empty".to_string(),
            ));
        }

        if self.nodes.is_empty() {
            return Ok(Vec::new());
        }

        // Get the entry point
        let entry_point_id = self.entry_point.expect("Entry point should exist");
        let similarity = self.compute_similarity(query, entry_point_id)?;
        let entry_dist = similarity.expect("Entry point should exist");

        // Start from the top level and traverse down
        let mut curr_node_id = entry_point_id;
        let mut curr_dist = entry_dist;

        // For each level, find the closest node
        for level_idx in (1..=self.max_level).rev() {
            let nearest = self.search_level(query, curr_node_id, curr_dist, level_idx, 1)?;
            if !nearest.is_empty() {
                curr_node_id = nearest[0].id;
                curr_dist = nearest[0].distance;
            }
        }

        // Perform the main search at level 0
        let ef = limit.max(self.config.ef);
        let candidates = self.search_level(query, curr_node_id, curr_dist, 0, ef)?;

        // Filter by threshold and limit - pre-allocate with a reasonable capacity
        let mut results = Vec::with_capacity(limit);
        for candidate in candidates {
            if candidate.distance >= threshold {
                results.push((candidate.id, candidate.distance));

                // Early exit once we have enough results
                if results.len() >= limit {
                    break;
                }
            }
        }

        Ok(results)
    }

    /// Returns the number of vectors in the index
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Checks if the index is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Gets a reference to a vector by its ID
    pub fn get(&self, id: Uuid) -> Option<&Vec<f32>> {
        self.nodes.get(&id).map(|node| &node.embedding)
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
        // Initialize visited set with a reasonable capacity to avoid reallocations
        let mut visited = std::collections::HashSet::with_capacity(ef * 2);
        visited.insert(entry_id);

        // Initialize candidate heap (max heap for candidates) with a reasonable capacity
        let mut candidates = BinaryHeap::with_capacity(ef * 2);
        candidates.push(Candidate {
            id: entry_id,
            distance: entry_dist,
        });

        // Initialize results heap (min heap for results) with capacity matching ef
        let mut results = BinaryHeap::with_capacity(ef + 1);
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
            // Avoid cloning the entire connections vector by borrowing and then collecting only the needed IDs
            let connected_ids = if let Some(node) = self.nodes.get(&curr.id) {
                if node.connections.len() > level {
                    // Borrow the connections and collect into a new Vec to avoid holding a reference too long
                    // This is memory efficient for temporary use in search
                    node.connections[level].to_vec()
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

        // Preallocate the result vector with exact capacity needed
        let result_capacity = results.len();
        let mut result_vec = Vec::with_capacity(result_capacity);

        // Convert results to a vector, reversing the negative distance
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
        // Pre-allocate with exact capacity needed
        let capacity = candidates.len().min(m);
        let mut selected = Vec::with_capacity(capacity);

        // If we have fewer candidates than M, use all of them
        if candidates.len() <= m {
            selected.extend(candidates.iter().map(|c| c.id));
            return selected;
        }

        // Take the top M candidates by similarity
        for (i, candidate) in candidates.iter().enumerate() {
            if i >= m {
                break;
            }
            selected.push(candidate.id);
        }

        selected
    }

    /// Generate a random level according to the HNSW distribution
    fn get_random_level(&mut self) -> usize {
        let r: f32 = self.rng.gen();
        (-r.ln() * self.config.level_multiplier) as usize
    }

    /// Compute similarity between a query vector and a node
    fn compute_similarity(&self, query: &[f32], node_id: Uuid) -> Result<Option<f32>> {
        if let Some(node) = self.nodes.get(&node_id) {
            Ok(cosine_similarity(query, &node.embedding))
        } else {
            Err(EngramDbError::Vector(format!(
                "Node not found in index: {}",
                node_id
            )))
        }
    }
}

// Implement the VectorSearchIndex trait for HnswIndex
impl VectorSearchIndex for HnswIndex {
    fn add(&mut self, node: &MemoryNode) -> Result<()> {
        // Get embeddings from the node
        if let Some(embeddings) = node.embeddings() {
            // Create a capacity-optimized vector to avoid over-allocation
            let mut vec = Vec::with_capacity(embeddings.len());
            vec.extend_from_slice(embeddings);

            // Call the struct's add method with node ID and embeddings
            self.add(node.id(), vec)
        } else {
            // This is a multi-vector node, which we can't handle
            Err(EngramDbError::Vector(format!(
                "Cannot add multi-vector embeddings to HNSW index: {}",
                node.id()
            )))
        }
    }

    fn remove(&mut self, id: Uuid) -> Result<()> {
        self.remove(id)
    }

    fn update(&mut self, node: &MemoryNode) -> Result<()> {
        // Get embeddings from the node
        if let Some(embeddings) = node.embeddings() {
            // Create a capacity-optimized vector to avoid over-allocation
            let mut vec = Vec::with_capacity(embeddings.len());
            vec.extend_from_slice(embeddings);

            // Call the struct's update method with node ID and embeddings
            self.update(node.id(), vec)
        } else {
            // This is a multi-vector node, which we can't handle
            Err(EngramDbError::Vector(format!(
                "Cannot update multi-vector embeddings in HNSW index: {}",
                node.id()
            )))
        }
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
    
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hnsw_basic() {
        let mut index = HnswIndex::new();

        // Create test vectors
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();

        let vec1 = vec![1.0, 0.0, 0.0];
        let vec2 = vec![0.0, 1.0, 0.0];
        let vec3 = vec![0.0, 0.0, 1.0];

        // Add to index
        index.add(id1, vec1.clone()).unwrap();
        index.add(id2, vec2.clone()).unwrap();
        index.add(id3, vec3.clone()).unwrap();

        assert_eq!(index.len(), 3);

        // Retrieve
        let retrieved1 = index.get(id1).unwrap();
        assert_eq!(retrieved1, &vec1);

        // Remove
        index.remove(id2).unwrap();
        assert_eq!(index.len(), 2);
        assert!(index.get(id2).is_none());

        // Update
        let vec1_updated = vec![0.5, 0.5, 0.0];
        index.update(id1, vec1_updated.clone()).unwrap();
        let retrieved1_updated = index.get(id1).unwrap();
        assert_eq!(retrieved1_updated, &vec1_updated);
    }

    #[test]
    fn test_hnsw_search() {
        let mut index = HnswIndex::with_config(HnswConfig {
            m: 8,
            ef_construction: 20,
            ef: 10,
            level_multiplier: 1.0,
            max_level: 4,
        });

        // Create test vectors
        let id1 = Uuid::new_v4();
        let id2 = Uuid::new_v4();
        let id3 = Uuid::new_v4();
        let id4 = Uuid::new_v4();

        let vec1 = vec![1.0, 0.0, 0.0]; // X-axis
        let vec2 = vec![0.7, 0.7, 0.0]; // 45 degrees in XY plane
        let vec3 = vec![0.0, 1.0, 0.0]; // Y-axis
        let vec4 = vec![0.0, 0.0, 1.0]; // Z-axis

        // Add to index
        index.add(id1, vec1).unwrap();
        index.add(id2, vec2).unwrap();
        index.add(id3, vec3).unwrap();
        index.add(id4, vec4).unwrap();

        // Search for something closer to the X-axis
        let query = vec![0.9, 0.1, 0.0];
        let results = index.search(&query, 2, 0.0).unwrap();

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, id1); // X-axis should be closest
        assert_eq!(results[1].0, id2); // 45 degrees should be second

        // Search with threshold
        let results = index.search(&query, 10, 0.9).unwrap();
        assert!(results.len() >= 1); // At least id1 should be above 0.9 similarity

        // Search with empty query
        let query = vec![];
        let result = index.search(&query, 10, 0.0);
        assert!(result.is_err());
    }

    #[test]
    fn test_hnsw_larger_dataset() {
        let mut index = HnswIndex::with_config(HnswConfig {
            m: 8,
            ef_construction: 20,
            ef: 10,
            level_multiplier: 1.0,
            max_level: 3,
        });

        // Create a larger set of random vectors
        let mut vectors = Vec::new();
        let mut ids = Vec::new();

        let dim = 10;
        let count = 100;

        let mut rng = rand::thread_rng();

        for _ in 0..count {
            let mut vec = Vec::with_capacity(dim);
            for _ in 0..dim {
                vec.push(rng.gen::<f32>());
            }

            // Normalize the vector
            let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();
            for x in &mut vec {
                *x /= norm;
            }

            vectors.push(vec);
            ids.push(Uuid::new_v4());
        }

        // Add all vectors to the index
        for (id, vec) in ids.iter().zip(vectors.iter()) {
            index.add(*id, vec.clone()).unwrap();
        }

        assert_eq!(index.len(), count);

        // Create a query vector
        let query = vectors[0].clone();

        // Search
        let results = index.search(&query, 5, 0.0).unwrap();

        // First result should be the vector itself with similarity very close to 1.0
        assert_eq!(results[0].0, ids[0]);
        assert!((results[0].1 - 1.0).abs() < 1e-5);
    }
}
