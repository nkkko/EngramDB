//! Memory node models for the EngramDB API

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Memory node create input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeCreateInput {
    /// Vector embedding
    pub vector: Vec<f32>,
    
    /// Attributes
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
    
    /// Connections
    #[serde(default)]
    pub connections: Vec<ConnectionInput>,
    
    /// Text content
    pub content: Option<String>,
}

/// Memory node update input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeUpdateInput {
    /// Vector embedding
    pub vector: Option<Vec<f32>>,
    
    /// Attributes
    pub attributes: Option<HashMap<String, serde_json::Value>>,
    
    /// Connections
    pub connections: Option<Vec<ConnectionInput>>,
    
    /// Text content
    pub content: Option<String>,
}

/// Memory node output model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeOutput {
    /// Node ID
    pub id: Uuid,
    
    /// Vector embedding (optional depending on request parameters)
    pub vector: Option<Vec<f32>>,
    
    /// Attributes
    pub attributes: HashMap<String, serde_json::Value>,
    
    /// Connections (optional depending on request parameters)
    pub connections: Option<Vec<ConnectionOutput>>,
    
    /// Created at timestamp
    pub created_at: DateTime<Utc>,
    
    /// Updated at timestamp
    pub updated_at: DateTime<Utc>,
    
    /// Text content
    pub content: Option<String>,
}

/// Memory node creation from content input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryNodeContentCreateInput {
    /// Text content
    pub content: String,
    
    /// Embedding model to use
    #[serde(default = "default_model")]
    pub model: String,
    
    /// Attributes
    #[serde(default)]
    pub attributes: HashMap<String, serde_json::Value>,
}

fn default_model() -> String {
    "default".to_string()
}

/// Connection input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInput {
    /// Target node ID
    pub target_id: Uuid,
    
    /// Relationship type
    pub type_name: String,
    
    /// Connection strength
    #[serde(default = "default_strength")]
    pub strength: f32,
    
    /// Custom type name (when type is "Custom")
    pub custom_type: Option<String>,
}

fn default_strength() -> f32 {
    1.0
}

/// Connection output model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionOutput {
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