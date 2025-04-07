use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::{AccessHistory, AttributeValue, Connection, TemporalLayer};

/// Represents a single memory node in the system
///
/// A MemoryNode is the fundamental unit of memory storage in the EngramDB system.
/// It combines vector embeddings for semantic representation, graph connections
/// for relational information, and temporal layers for time-based versioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNode {
    /// Unique identifier for the memory node
    id: Uuid,

    /// Vector embeddings representing the semantic content
    embeddings: Vec<f32>,

    /// Graph connections to other memory nodes
    connections: Vec<Connection>,

    /// Time-aware versions of this memory
    temporal_layers: Vec<TemporalLayer>,

    /// Flexible schema allowing for arbitrary attributes
    attributes: HashMap<String, AttributeValue>,

    /// Timestamp when this memory was created
    creation_timestamp: u64,

    /// Usage statistics for optimizing access patterns
    access_patterns: AccessHistory,
}

impl MemoryNode {
    /// Creates a new memory node with the given embeddings
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Vector representation of the memory content
    ///
    /// # Returns
    ///
    /// A new MemoryNode with a generated UUID and initialized fields
    pub fn new(embeddings: Vec<f32>) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: Uuid::new_v4(),
            embeddings,
            connections: Vec::new(),
            temporal_layers: Vec::new(),
            attributes: HashMap::new(),
            creation_timestamp: now,
            access_patterns: AccessHistory::new(),
        }
    }
    
    /// Creates a new memory node from text content using the provided embedding service
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to encode into embeddings
    /// * `embedding_service` - The embedding service to use
    /// * `category` - Optional category for the content
    ///
    /// # Returns
    ///
    /// A new MemoryNode with embeddings generated from the text
    #[cfg(feature = "embeddings")]
    pub fn from_text(
        text: &str,
        embedding_service: &crate::embeddings::EmbeddingService,
        category: Option<&str>,
    ) -> Result<Self, crate::embeddings::EmbeddingError> {
        // Generate embeddings for the text
        let embeddings = embedding_service.generate_for_document(text, category)?;
        
        // Create the memory node
        let mut node = Self::new(embeddings);
        
        // Store the text content as an attribute
        node.set_attribute("content".to_string(), AttributeValue::String(text.to_string()));
        
        // Store the category if provided
        if let Some(cat) = category {
            node.set_attribute("category".to_string(), AttributeValue::String(cat.to_string()));
        }
        
        Ok(node)
    }
    
    /// Creates a new memory node with random embeddings (for testing)
    ///
    /// # Arguments
    ///
    /// * `dimensions` - The dimensionality of the random embeddings
    ///
    /// # Returns
    ///
    /// A new MemoryNode with random embeddings
    #[cfg(feature = "embeddings")]
    pub fn with_random_embeddings(dimensions: usize) -> Self {
        use crate::embeddings::mock::generate_random_embedding;
        Self::new(generate_random_embedding(dimensions))
    }

    /// Test-only method to create a node with a specific ID and timestamp
    /// This method should only be used in tests
    #[cfg(test)]
    pub fn test_with_id_and_timestamp(id: Uuid, embeddings: Vec<f32>, timestamp: u64) -> Self {
        Self {
            id,
            embeddings,
            connections: Vec::new(),
            temporal_layers: Vec::new(),
            attributes: HashMap::new(),
            creation_timestamp: timestamp,
            access_patterns: AccessHistory::new(),
        }
    }

    /// Returns the unique identifier of this memory node
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returns a reference to the vector embeddings
    pub fn embeddings(&self) -> &[f32] {
        &self.embeddings
    }

    /// Sets the embeddings to a new value
    pub fn set_embeddings(&mut self, embeddings: Vec<f32>) {
        self.embeddings = embeddings;
    }

    /// Returns a reference to the connections
    pub fn connections(&self) -> &[Connection] {
        &self.connections
    }

    /// Adds a new connection to another memory node
    pub fn add_connection(&mut self, connection: Connection) {
        self.connections.push(connection);
    }

    /// Returns a reference to the temporal layers
    pub fn temporal_layers(&self) -> &[TemporalLayer] {
        &self.temporal_layers
    }

    /// Adds a new temporal layer
    pub fn add_temporal_layer(&mut self, layer: TemporalLayer) {
        self.temporal_layers.push(layer);
    }

    /// Returns a reference to the attributes
    pub fn attributes(&self) -> &HashMap<String, AttributeValue> {
        &self.attributes
    }

    /// Gets an attribute by key
    pub fn get_attribute(&self, key: &str) -> Option<&AttributeValue> {
        self.attributes.get(key)
    }

    /// Sets an attribute
    pub fn set_attribute(&mut self, key: String, value: AttributeValue) {
        self.attributes.insert(key, value);
    }

    /// Returns the creation timestamp
    pub fn creation_timestamp(&self) -> u64 {
        self.creation_timestamp
    }

    /// Returns a reference to the access patterns
    pub fn access_patterns(&self) -> &AccessHistory {
        &self.access_patterns
    }

    /// Records an access to this memory node
    pub fn record_access(&mut self) {
        self.access_patterns.record_access();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_node_creation() {
        let embeddings = vec![0.1, 0.2, 0.3];
        let node = MemoryNode::new(embeddings.clone());

        assert_eq!(node.embeddings(), &embeddings);
        assert!(node.connections().is_empty());
        assert!(node.temporal_layers().is_empty());
        assert!(node.attributes().is_empty());
        assert!(node.creation_timestamp() > 0);
    }

    #[test]
    fn test_attributes() {
        let mut node = MemoryNode::new(vec![0.1, 0.2, 0.3]);

        // Add some attributes
        node.set_attribute(
            "name".to_string(),
            AttributeValue::String("test memory".to_string()),
        );
        node.set_attribute("importance".to_string(), AttributeValue::Float(0.8));

        // Check attributes
        assert_eq!(
            node.get_attribute("name"),
            Some(&AttributeValue::String("test memory".to_string()))
        );
        assert_eq!(
            node.get_attribute("importance"),
            Some(&AttributeValue::Float(0.8))
        );
        assert_eq!(node.get_attribute("non_existent"), None);
    }
}
