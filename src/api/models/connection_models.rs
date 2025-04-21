//! Connection models for the EngramDB API

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Create connection input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateConnectionInput {
    /// Target node ID
    pub target_id: Uuid,
    
    /// Relationship type
    pub type_name: String,
    
    /// Connection strength
    #[serde(default = "default_strength")]
    pub strength: f32,
    
    /// Custom type name (when type is "Custom")
    pub custom_type: Option<String>,
    
    /// Whether to create a bidirectional connection
    #[serde(default)]
    pub bidirectional: bool,
}

fn default_strength() -> f32 {
    1.0
}

/// Connection details output model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionDetails {
    /// Source node ID
    pub source_id: Uuid,
    
    /// Target node ID
    pub target_id: Uuid,
    
    /// Relationship type
    pub type_name: String,
    
    /// Connection strength
    pub strength: f32,
    
    /// Custom type name (when type is "Custom")
    pub custom_type: Option<String>,
}