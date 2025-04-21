//! Search-related models for the EngramDB API

use crate::api::models::memory_node_models::MemoryNodeOutput;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Search query input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQueryInput {
    /// Vector to search for
    pub vector: Option<Vec<f32>>,

    /// Text content to convert to embedding
    pub content: Option<String>,

    /// Embedding model to use for content
    #[serde(default = "default_model")]
    pub model: String,

    /// Attribute filters
    #[serde(default)]
    pub filters: Vec<AttributeFilterInput>,

    /// Maximum number of results
    #[serde(default = "default_limit")]
    pub limit: usize,

    /// Minimum similarity threshold
    #[serde(default)]
    pub threshold: f32,

    /// Whether to include vector embeddings in results
    #[serde(default)]
    pub include_vectors: bool,

    /// Whether to include connections in results
    #[serde(default)]
    pub include_connections: bool,
}

fn default_model() -> String {
    "default".to_string()
}

fn default_limit() -> usize {
    10
}

/// Attribute filter input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributeFilterInput {
    /// Attribute field name
    pub field: String,

    /// Filter operation
    pub operation: String,

    /// Filter value
    pub value: Option<serde_json::Value>,
}

/// Search result item model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResultItem {
    /// The memory node
    pub node: MemoryNodeOutput,

    /// Similarity score
    pub similarity: f32,
}

/// Search results output model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResults {
    /// Result items
    pub results: Vec<SearchResultItem>,

    /// Total count (may be higher than results.len() if limited)
    pub total: usize,
}
