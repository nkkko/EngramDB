//! Embedding-related models for the EngramDB API

use serde::{Deserialize, Serialize};

/// Embedding model information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingModelInfo {
    /// Model ID
    pub id: String,

    /// Model name
    pub name: String,

    /// Vector dimensions
    pub dimensions: usize,

    /// Model description
    pub description: String,

    /// Provider name
    pub provider: String,

    /// Model type (single_vector or multi_vector)
    pub model_type: String,
}

/// Generate embedding input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateEmbeddingInput {
    /// Text content
    pub content: String,

    /// Model ID
    #[serde(default = "default_model")]
    pub model: String,
}

fn default_model() -> String {
    "default".to_string()
}

/// Generated embedding output model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedEmbedding {
    /// Vector embedding
    pub vector: Vec<f32>,

    /// Model used
    pub model: String,

    /// Vector dimensions
    pub dimensions: usize,
}
