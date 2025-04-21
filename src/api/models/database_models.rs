//! Database-related models for the EngramDB API

use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use crate::storage::StorageType;
use crate::vector::{VectorAlgorithm, HnswConfig};

/// Database configuration input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfigInput {
    /// Storage type
    #[serde(default = "default_storage_type")]
    pub storage_type: String,
    
    /// Storage path (for file-based databases)
    pub storage_path: Option<String>,
    
    /// Cache size
    #[serde(default = "default_cache_size")]
    pub cache_size: usize,
    
    /// Vector algorithm
    #[serde(default = "default_vector_algorithm")]
    pub vector_algorithm: String,
    
    /// HNSW configuration
    pub hnsw_config: Option<HnswConfigInput>,
}

fn default_storage_type() -> String {
    "MultiFile".to_string()
}

fn default_cache_size() -> usize {
    100
}

fn default_vector_algorithm() -> String {
    "HNSW".to_string()
}

impl Default for DatabaseConfigInput {
    fn default() -> Self {
        Self {
            storage_type: default_storage_type(),
            storage_path: None,
            cache_size: default_cache_size(),
            vector_algorithm: default_vector_algorithm(),
            hnsw_config: None,
        }
    }
}

/// HNSW configuration input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HnswConfigInput {
    /// Maximum number of connections per node
    #[serde(default = "default_m")]
    pub m: usize,
    
    /// Size of dynamic candidate list during construction
    #[serde(default = "default_ef_construction")]
    pub ef_construction: usize,
    
    /// Size of dynamic candidate list during search
    #[serde(default = "default_ef")]
    pub ef: usize,
    
    /// Level multiplier
    #[serde(default = "default_level_multiplier")]
    pub level_multiplier: f32,
    
    /// Maximum level
    #[serde(default = "default_max_level")]
    pub max_level: usize,
}

fn default_m() -> usize {
    16
}

fn default_ef_construction() -> usize {
    100
}

fn default_ef() -> usize {
    10
}

fn default_level_multiplier() -> f32 {
    1.0
}

fn default_max_level() -> usize {
    16
}

impl From<HnswConfigInput> for HnswConfig {
    fn from(input: HnswConfigInput) -> Self {
        Self {
            m: input.m,
            ef_construction: input.ef_construction,
            ef: input.ef,
            level_multiplier: input.level_multiplier,
            max_level: input.max_level,
        }
    }
}

/// Database creation input model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreateDatabaseInput {
    /// Database name
    pub name: String,
    
    /// Database configuration
    #[serde(default)]
    pub config: DatabaseConfigInput,
}

/// Database information output model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseInfo {
    /// Database ID
    pub id: String,
    
    /// Database name
    pub name: String,
    
    /// Storage type
    pub storage_type: String,
    
    /// Node count
    pub node_count: usize,
    
    /// Created at timestamp
    pub created_at: DateTime<Utc>,
    
    /// Configuration
    pub config: DatabaseConfigInput,
}

/// Convert a storage type string to StorageType enum
pub fn parse_storage_type(storage_type: &str) -> StorageType {
    match storage_type.to_lowercase().as_str() {
        "memory" => StorageType::Memory,
        "singlefile" | "single_file" | "single-file" => StorageType::SingleFile,
        _ => StorageType::MultiFile, // Default to multi-file
    }
}

/// Convert a vector algorithm string to VectorAlgorithm enum
pub fn parse_vector_algorithm(algorithm: &str) -> VectorAlgorithm {
    match algorithm.to_uppercase().as_str() {
        "LINEAR" => VectorAlgorithm::Linear,
        _ => VectorAlgorithm::HNSW, // Default to HNSW
    }
}