use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

use super::AttributeValue;

/// Represents a temporal version of a memory node
///
/// TemporalLayer allows the system to track how memories change over time,
/// maintaining both the content and metadata about the evolution of knowledge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalLayer {
    /// Unique identifier for this temporal layer
    id: Uuid,

    /// Timestamp when this layer was created
    timestamp: u64,

    /// The version of the embedding vector at this point in time
    embeddings: Option<Vec<f32>>,

    /// The version of attributes at this point in time
    attributes: Option<HashMap<String, AttributeValue>>,

    /// Reason why this temporal layer was created
    reason: String,
}

impl TemporalLayer {
    /// Creates a new temporal layer
    ///
    /// # Arguments
    ///
    /// * `embeddings` - Optional vector embeddings for this layer
    /// * `attributes` - Optional attribute map for this layer
    /// * `reason` - The reason why this layer was created
    ///
    /// # Returns
    ///
    /// A new TemporalLayer with the specified content
    pub fn new(
        embeddings: Option<Vec<f32>>,
        attributes: Option<HashMap<String, AttributeValue>>,
        reason: String,
    ) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            id: Uuid::new_v4(),
            timestamp: now,
            embeddings,
            attributes,
            reason,
        }
    }

    /// Returns the unique identifier of this temporal layer
    pub fn id(&self) -> Uuid {
        self.id
    }

    /// Returns the timestamp when this layer was created
    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }

    /// Returns a reference to the embeddings, if any
    pub fn embeddings(&self) -> Option<&[f32]> {
        self.embeddings.as_deref()
    }

    /// Returns a reference to the attributes, if any
    pub fn attributes(&self) -> Option<&HashMap<String, AttributeValue>> {
        self.attributes.as_ref()
    }

    /// Returns the reason why this layer was created
    pub fn reason(&self) -> &str {
        &self.reason
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_layer_creation() {
        let embeddings = Some(vec![0.1, 0.2, 0.3]);
        let mut attributes = HashMap::new();
        attributes.insert("importance".to_string(), AttributeValue::Float(0.8));

        let layer = TemporalLayer::new(
            embeddings.clone(),
            Some(attributes.clone()),
            "Initial creation".to_string(),
        );

        assert_eq!(layer.embeddings(), embeddings.as_deref());
        assert_eq!(layer.attributes(), Some(&attributes));
        assert_eq!(layer.reason(), "Initial creation");
        assert!(layer.timestamp() > 0);
    }

    #[test]
    fn test_temporal_layer_partial_data() {
        // Test with only embeddings
        let embeddings = Some(vec![0.1, 0.2, 0.3]);
        let layer = TemporalLayer::new(embeddings.clone(), None, "Embedding update".to_string());

        assert_eq!(layer.embeddings(), embeddings.as_deref());
        assert_eq!(layer.attributes(), None);

        // Test with only attributes
        let mut attributes = HashMap::new();
        attributes.insert("importance".to_string(), AttributeValue::Float(0.8));

        let layer = TemporalLayer::new(
            None,
            Some(attributes.clone()),
            "Attribute update".to_string(),
        );

        assert_eq!(layer.embeddings(), None);
        assert_eq!(layer.attributes(), Some(&attributes));
    }
}
