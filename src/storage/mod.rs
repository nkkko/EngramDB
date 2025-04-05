//! Storage engine for persisting memory nodes
//!
//! This module handles the persistence and retrieval of memory nodes,
//! providing CRUD operations for the memory database.

mod file_storage;
mod memory_storage;

pub use file_storage::FileStorageEngine;
pub use memory_storage::MemoryStorageEngine;

use crate::core::MemoryNode;
use crate::Result;
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
}
