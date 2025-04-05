use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Represents a connection between memory nodes in the graph
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Connection {
    /// ID of the target memory node
    target_id: Uuid,

    /// Type of relationship between the nodes
    relationship_type: RelationshipType,

    /// Strength of the connection (0.0 to 1.0)
    strength: f32,

    /// When this connection was created
    creation_timestamp: u64,
}

/// Types of relationships that can exist between memory nodes
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationshipType {
    /// Simple association between memories
    Association,

    /// Causal relationship (this memory caused the target)
    Causation,

    /// This memory is a part of the target memory
    PartOf,

    /// This memory contains the target memory
    Contains,

    /// Temporal sequence relationship
    Sequence,

    /// Custom relationship with a string label
    Custom(String),
}

impl Connection {
    /// Creates a new connection to another memory node
    ///
    /// # Arguments
    ///
    /// * `target_id` - The UUID of the target memory node
    /// * `relationship_type` - The type of relationship
    /// * `strength` - The strength of the connection (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A new Connection object
    pub fn new(target_id: Uuid, relationship_type: RelationshipType, strength: f32) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            target_id,
            relationship_type,
            strength: strength.clamp(0.0, 1.0),
            creation_timestamp: now,
        }
    }

    /// Returns the target memory node ID
    pub fn target_id(&self) -> Uuid {
        self.target_id
    }

    /// Returns the relationship type
    pub fn relationship_type(&self) -> &RelationshipType {
        &self.relationship_type
    }

    /// Returns the strength of the connection
    pub fn strength(&self) -> f32 {
        self.strength
    }

    /// Sets the strength to a new value (clamped between 0.0 and 1.0)
    pub fn set_strength(&mut self, strength: f32) {
        self.strength = strength.clamp(0.0, 1.0);
    }

    /// Returns the creation timestamp
    pub fn creation_timestamp(&self) -> u64 {
        self.creation_timestamp
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_connection_creation() {
        let target_id = Uuid::new_v4();
        let connection = Connection::new(target_id, RelationshipType::Association, 0.75);

        assert_eq!(connection.target_id(), target_id);
        assert_eq!(
            connection.relationship_type(),
            &RelationshipType::Association
        );
        assert_eq!(connection.strength(), 0.75);
        assert!(connection.creation_timestamp() > 0);
    }

    #[test]
    fn test_strength_clamping() {
        // Test with a value too high
        let connection = Connection::new(Uuid::new_v4(), RelationshipType::Causation, 1.5);
        assert_eq!(connection.strength(), 1.0);

        // Test with a value too low
        let connection = Connection::new(Uuid::new_v4(), RelationshipType::Causation, -0.5);
        assert_eq!(connection.strength(), 0.0);
    }
}
