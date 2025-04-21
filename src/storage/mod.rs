//! Storage engine for persisting memory nodes
//!
//! This module handles the persistence and retrieval of memory nodes,
//! providing CRUD operations for the memory database.

mod file_storage;
mod memory_storage;
mod single_file_storage;

pub use file_storage::FileStorageEngine;
pub use memory_storage::MemoryStorageEngine;
pub use single_file_storage::SingleFileStorageEngine;

use crate::core::MemoryNode;
use crate::database::StorageType;
use crate::Result;
use std::path::Path;
use uuid::Uuid;

/// Trait defining the interface for storage engines
pub trait StorageEngine: Send {
    /// Saves a memory node to storage
    ///
    /// # Arguments
    ///
    /// * `node` - The memory node to save
    ///
    /// # Returns
    ///
    /// A Result indicating success or the specific error that occurred
    fn save(&mut self, node: &MemoryNode) -> Result<()>;

    /// Loads a memory node by its ID
    ///
    /// # Arguments
    ///
    /// * `id` - The UUID of the memory node to load
    ///
    /// # Returns
    ///
    /// The requested memory node if found, otherwise an error
    fn load(&self, id: Uuid) -> Result<MemoryNode>;

    /// Deletes a memory node by its ID
    ///
    /// # Arguments
    ///
    /// * `id` - The UUID of the memory node to delete
    ///
    /// # Returns
    ///
    /// A Result indicating success or the specific error that occurred
    fn delete(&mut self, id: Uuid) -> Result<()>;

    /// Lists all memory node IDs in storage
    ///
    /// # Returns
    ///
    /// A vector of UUIDs for all stored memory nodes
    fn list_all(&self) -> Result<Vec<Uuid>>;

    /// Gets the storage type of this engine
    ///
    /// # Returns
    ///
    /// The storage type enum
    fn get_type(&self) -> StorageType;

    /// Gets the storage path of this engine if applicable
    ///
    /// # Returns
    ///
    /// An Option containing the Path if this is a file-based storage engine,
    /// or None if this is an in-memory storage engine
    fn get_path(&self) -> Option<&Path>;
}
