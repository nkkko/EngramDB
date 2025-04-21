use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use uuid::Uuid;

use super::StorageEngine;
use crate::core::MemoryNode;
use crate::error::EngramDbError;
use crate::Result;

/// An in-memory storage engine for memory nodes
///
/// This storage engine keeps all memory nodes in memory,
/// making it fast but volatile (data is lost when the program exits).
/// It's useful for testing, development, and scenarios where persistence
/// is not required.
#[derive(Clone, Default)]
pub struct MemoryStorageEngine {
    /// Internal storage of memory nodes, indexed by UUID
    store: Arc<RwLock<HashMap<Uuid, MemoryNode>>>,
}

impl MemoryStorageEngine {
    /// Creates a new empty in-memory storage engine
    pub fn new() -> Self {
        Self {
            store: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Returns the number of memory nodes in the store
    pub fn len(&self) -> usize {
        self.store.read().unwrap().len()
    }

    /// Checks if the store is empty
    pub fn is_empty(&self) -> bool {
        self.store.read().unwrap().is_empty()
    }

    /// Clears all memory nodes from the store
    pub fn clear(&mut self) {
        self.store.write().unwrap().clear();
    }
}

impl StorageEngine for MemoryStorageEngine {
    fn save(&mut self, node: &MemoryNode) -> Result<()> {
        let mut store = self.store.write().unwrap();
        store.insert(node.id(), node.clone());
        Ok(())
    }

    fn load(&self, id: Uuid) -> Result<MemoryNode> {
        let store = self.store.read().unwrap();

        store
            .get(&id)
            .cloned()
            .ok_or_else(|| EngramDbError::Storage(format!("Memory node not found: {}", id)))
    }

    fn delete(&mut self, id: Uuid) -> Result<()> {
        let mut store = self.store.write().unwrap();

        if store.remove(&id).is_none() {
            return Err(EngramDbError::Storage(format!(
                "Memory node not found: {}",
                id
            )));
        }

        Ok(())
    }

    fn list_all(&self) -> Result<Vec<Uuid>> {
        let store = self.store.read().unwrap();
        let ids = store.keys().cloned().collect();
        Ok(ids)
    }

    fn get_type(&self) -> crate::database::StorageType {
        crate::database::StorageType::Memory
    }

    fn get_path(&self) -> Option<&std::path::Path> {
        None // Memory storage has no path
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_storage_crud() {
        let mut storage = MemoryStorageEngine::new();

        // Create a test memory node
        let node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        let id = node.id();

        // Test save
        storage.save(&node).unwrap();
        assert_eq!(storage.len(), 1);

        // Test load
        let loaded_node = storage.load(id).unwrap();
        assert_eq!(loaded_node.id(), node.id());
        assert_eq!(loaded_node.embeddings(), node.embeddings());

        // Test list_all
        let ids = storage.list_all().unwrap();
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], id);

        // Test delete
        storage.delete(id).unwrap();
        assert!(storage.is_empty());

        // Verify it's gone
        let result = storage.load(id);
        assert!(result.is_err());

        let ids = storage.list_all().unwrap();
        assert_eq!(ids.len(), 0);
    }

    #[test]
    fn test_memory_storage_clear() {
        let mut storage = MemoryStorageEngine::new();

        // Add some nodes
        for _ in 0..5 {
            let node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
            storage.save(&node).unwrap();
        }

        assert_eq!(storage.len(), 5);

        // Clear storage
        storage.clear();

        assert!(storage.is_empty());
        assert_eq!(storage.list_all().unwrap().len(), 0);
    }

    #[test]
    fn test_memory_storage_clone() {
        let mut storage1 = MemoryStorageEngine::new();

        // Add a node
        let node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        let id = node.id();
        storage1.save(&node).unwrap();

        // Clone the storage
        let storage2 = storage1.clone();

        // Both should have the same content
        assert_eq!(storage1.len(), 1);
        assert_eq!(storage2.len(), 1);

        // Load from the clone
        let loaded_node = storage2.load(id).unwrap();
        assert_eq!(loaded_node.id(), id);

        // Modify the original
        storage1.delete(id).unwrap();

        // The clone should reflect the change (shared state)
        assert!(storage1.is_empty());
        assert!(storage2.is_empty());
    }
}
