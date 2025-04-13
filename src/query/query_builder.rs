use uuid::Uuid;

use crate::core::MemoryNode;
use crate::error::EngramDbError;
use crate::vector::VectorSearchIndex;
use crate::Result;

use super::filter::{AttributeFilter, TemporalFilter};

/// A builder for constructing and executing memory queries
///
/// This struct allows for a fluent interface to build complex queries
/// combining vector similarity, attribute filtering, and temporal constraints.
pub struct QueryBuilder {
    /// Vector query for similarity search
    query_vector: Option<Vec<f32>>,

    /// Similarity threshold (0.0 to 1.0)
    similarity_threshold: f32,

    /// Maximum number of results to return
    limit: usize,

    /// Attribute filters to apply
    attribute_filters: Vec<AttributeFilter>,

    /// Temporal filters to apply
    temporal_filters: Vec<TemporalFilter>,

    /// IDs to include as potential matches
    include_ids: Option<Vec<Uuid>>,

    /// IDs to exclude from results
    exclude_ids: Vec<Uuid>,
}

impl Default for QueryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryBuilder {
    /// Creates a new empty query builder with default settings
    pub fn new() -> Self {
        Self {
            query_vector: None,
            similarity_threshold: 0.0,
            limit: 100,
            attribute_filters: Vec::new(),
            temporal_filters: Vec::new(),
            include_ids: None,
            exclude_ids: Vec::new(),
        }
    }

    /// Sets the query vector for similarity search
    ///
    /// # Arguments
    ///
    /// * `vector` - The vector to compare against
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_vector(mut self, vector: Vec<f32>) -> Self {
        self.query_vector = Some(vector);
        self
    }

    /// Sets the similarity threshold
    ///
    /// # Arguments
    ///
    /// * `threshold` - Minimum similarity score (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.similarity_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Sets the maximum number of results to return
    ///
    /// # Arguments
    ///
    /// * `limit` - Maximum number of results
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Adds an attribute filter to the query
    ///
    /// # Arguments
    ///
    /// * `filter` - The attribute filter to apply
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_attribute_filter(mut self, filter: AttributeFilter) -> Self {
        self.attribute_filters.push(filter);
        self
    }

    /// Adds a temporal filter to the query
    ///
    /// # Arguments
    ///
    /// * `filter` - The temporal filter to apply
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_temporal_filter(mut self, filter: TemporalFilter) -> Self {
        self.temporal_filters.push(filter);
        self
    }

    /// Restricts the query to only consider the specified IDs
    ///
    /// # Arguments
    ///
    /// * `ids` - The IDs to include as potential matches
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_include_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.include_ids = Some(ids);
        self
    }

    /// Adds IDs to exclude from the results
    ///
    /// # Arguments
    ///
    /// * `ids` - The IDs to exclude
    ///
    /// # Returns
    ///
    /// Self for method chaining
    pub fn with_exclude_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.exclude_ids.extend(ids);
        self
    }

    /// Executes the query against a vector index and a set of memory nodes
    ///
    /// # Arguments
    ///
    /// * `vector_index` - The vector index to search
    /// * `memory_nodes` - A function that can retrieve memory nodes by ID
    ///
    /// # Returns
    ///
    /// A vector of memory nodes matching the query, sorted by relevance
    pub fn execute<F>(
        &self,
        vector_index: &(dyn VectorSearchIndex + Send + Sync),
        mut memory_nodes: F,
    ) -> Result<Vec<MemoryNode>>
    where
        F: FnMut(Uuid) -> Result<MemoryNode>,
    {
        // Determine candidate IDs
        let candidate_ids = if let Some(query_vector) = &self.query_vector {
            // If we have a query vector, use vector search
            let search_results =
                vector_index.search(query_vector, self.limit, self.similarity_threshold)?;

            // Extract IDs from search results
            search_results.into_iter().map(|(id, _)| id).collect()
        } else if let Some(include_ids) = &self.include_ids {
            // If no query vector but we have include_ids, use those
            include_ids.clone()
        } else {
            // We need either a query vector or include_ids
            return Err(EngramDbError::Query(
                "Query must specify either a vector or include_ids".to_string(),
            ));
        };

        // Apply exclusions
        let candidate_ids: Vec<Uuid> = candidate_ids
            .into_iter()
            .filter(|id| !self.exclude_ids.contains(id))
            .collect();

        // Fetch and filter memory nodes
        let mut result_nodes = Vec::new();

        for id in candidate_ids {
            // Fetch the memory node
            let node = memory_nodes(id)?;

            // Apply attribute filters
            let passes_attribute_filters = self
                .attribute_filters
                .iter()
                .all(|filter| filter.apply(&node));
            if !passes_attribute_filters {
                continue;
            }

            // Apply temporal filters
            let passes_temporal_filters = self
                .temporal_filters
                .iter()
                .all(|filter| filter.apply(&node));
            if !passes_temporal_filters {
                continue;
            }

            // This node passed all filters
            result_nodes.push(node);

            // Check if we've reached the limit
            if result_nodes.len() >= self.limit {
                break;
            }
        }

        Ok(result_nodes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::AttributeValue;
    use crate::vector::{create_vector_index, VectorIndexConfig};
    use std::collections::HashMap;

    // Helper to create a test memory node with specific attributes
    fn create_test_node(
        id: Uuid,
        embeddings: Vec<f32>,
        attributes: HashMap<String, AttributeValue>,
        timestamp: u64,
    ) -> MemoryNode {
        // Create the node with ID and timestamp
        let mut node = MemoryNode::test_with_id_and_timestamp(id, embeddings, timestamp);

        // Set the attributes
        for (key, value) in attributes {
            node.set_attribute(key, value);
        }

        node
    }

    #[test]
    fn test_query_builder_simple_vector_query() {
        // Create a fixed UUID for testing to ensure repeatability
        let id1 = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let id2 = Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
        let id3 = Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();

        // Create a vector index
        let mut vector_index = create_vector_index(&VectorIndexConfig::default());

        // Set up test node map
        let mut node_map = HashMap::new();

        // Create nodes with test embeddings

        // Node 1: [1, 0, 0]
        let node1 = create_test_node(id1, vec![1.0, 0.0, 0.0], HashMap::new(), 100);

        // Node 2: [0, 1, 0]
        let node2 = create_test_node(id2, vec![0.0, 1.0, 0.0], HashMap::new(), 200);

        // Node 3: [0, 0, 1]
        let node3 = create_test_node(id3, vec![0.0, 0.0, 1.0], HashMap::new(), 300);

        // Add nodes to the vector index
        vector_index.add(&node1).unwrap();
        vector_index.add(&node2).unwrap();
        vector_index.add(&node3).unwrap();

        // Add nodes to our test map
        node_map.insert(id1, node1.clone());
        node_map.insert(id2, node2.clone());
        node_map.insert(id3, node3.clone());

        // Create a query for vectors similar to [1, 0, 0]
        let query = QueryBuilder::new()
            .with_vector(vec![1.0, 0.0, 0.0])
            .with_limit(2);

        // Execute the query with a custom node loader
        let results = query
            .execute(vector_index.as_ref(), |id| {
                // Look up the node in our test map
                match node_map.get(&id) {
                    Some(node) => Ok(node.clone()),
                    None => Err(EngramDbError::Storage(format!("Node not found: {}", id))),
                }
            })
            .unwrap();

        // Since we're using cosine similarity, both node1 (aligned with query)
        // and node2 (90 degrees to query) will be returned with <= 2 limit
        assert!(results.len() <= 2);
        // Node1 should be first since it's most similar
        assert_eq!(results[0].id(), id1);
    }

    #[test]
    fn test_query_builder_combined_filters() {
        // Create fixed UUIDs for testing
        let id1 = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let id2 = Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
        let id3 = Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();

        // Create a vector index
        let mut vector_index = create_vector_index(&VectorIndexConfig::default());

        // Test node map
        let mut node_map = HashMap::new();

        // Node 1: [1, 0, 0] with attributes and timestamp 100
        let mut attrs1 = HashMap::new();
        attrs1.insert("importance".to_string(), AttributeValue::Float(0.8));
        attrs1.insert(
            "type".to_string(),
            AttributeValue::String("memory".to_string()),
        );

        let node1 = create_test_node(id1, vec![1.0, 0.0, 0.0], attrs1, 100);

        // Node 2: [0.7, 0.7, 0] with attributes and timestamp 200
        let mut attrs2 = HashMap::new();
        attrs2.insert("importance".to_string(), AttributeValue::Float(0.5));
        attrs2.insert(
            "type".to_string(),
            AttributeValue::String("memory".to_string()),
        );

        let node2 = create_test_node(id2, vec![0.7, 0.7, 0.0], attrs2, 200);

        // Node 3: [0, 0, 1] with attributes and timestamp 300
        let mut attrs3 = HashMap::new();
        attrs3.insert("importance".to_string(), AttributeValue::Float(0.9));
        attrs3.insert(
            "type".to_string(),
            AttributeValue::String("fact".to_string()),
        );

        let node3 = create_test_node(id3, vec![0.0, 0.0, 1.0], attrs3, 300);

        // Add to vector index
        vector_index.add(&node1).unwrap();
        vector_index.add(&node2).unwrap();
        vector_index.add(&node3).unwrap();

        // Add to test map
        node_map.insert(id1, node1.clone());
        node_map.insert(id2, node2.clone());
        node_map.insert(id3, node3.clone());

        // Create filters

        // Type filter: "type" = "memory"
        let type_filter = AttributeFilter::equals(
            "type".to_string(),
            AttributeValue::String("memory".to_string()),
        );

        // Importance filter: "importance" > 0.7
        let importance_filter =
            AttributeFilter::greater_than("importance".to_string(), AttributeValue::Float(0.7));

        // Temporal filter: timestamp < 200
        let temporal_filter = TemporalFilter::before(200);

        // Create a query combining all filters with vector similarity
        let query = QueryBuilder::new()
            .with_vector(vec![0.9, 0.1, 0.0]) // Similar to [1, 0, 0]
            .with_attribute_filter(type_filter)
            .with_attribute_filter(importance_filter)
            .with_temporal_filter(temporal_filter);

        // Execute the query
        let results = query
            .execute(vector_index.as_ref(), |id| match node_map.get(&id) {
                Some(node) => Ok(node.clone()),
                None => Err(EngramDbError::Storage(format!("Node not found: {}", id))),
            })
            .unwrap();

        // We should get only node1:
        // - Most similar to query vector
        // - Has type "memory"
        // - Has importance 0.8 (> 0.7)
        // - Has timestamp 100 (< 200)
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id(), id1);
    }
}
