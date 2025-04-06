//! EngramDB: Engram Database
//!
//! A specialized database system designed for agent memory management.
//! This library provides efficient storage, retrieval, and querying of
//! agent memories using a unified memory representation model.

pub mod core;
pub mod database;
pub mod query;
pub mod storage;
pub mod utils;
pub mod vector;

pub use crate::core::MemoryNode;
pub use crate::core::connection::RelationshipType;
pub use crate::database::Database;
pub use crate::database::DatabaseConfig;
pub use crate::database::StorageType;
pub use crate::query::QueryBuilder;
pub use crate::storage::StorageEngine;

/// Error types for the EngramDB system
pub mod error {
    use thiserror::Error;

    /// Represents all possible errors that can occur in the EngramDB system
    #[derive(Error, Debug)]
    pub enum EngramDbError {
        /// Error related to storage operations
        #[error("Storage error: {0}")]
        Storage(String),

        /// Error related to query operations
        #[error("Query error: {0}")]
        Query(String),

        /// Error related to vector operations
        #[error("Vector error: {0}")]
        Vector(String),

        /// Error related to serialization/deserialization
        #[error("Serialization error: {0}")]
        Serialization(String),

        /// Error related to validation
        #[error("Validation error: {0}")]
        Validation(String),

        /// Generic errors that don't fit into other categories
        #[error("Other error: {0}")]
        Other(String),
    }
}

/// Result type for EngramDB operations
pub type Result<T> = std::result::Result<T, error::EngramDbError>;
