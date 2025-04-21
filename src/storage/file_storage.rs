use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use super::StorageEngine;
use crate::core::MemoryNode;
use crate::error::EngramDbError;
use crate::Result;

/// A simple file-based storage engine for memory nodes
///
/// This storage engine saves each memory node as a separate file
/// in a specified directory structure.
pub struct FileStorageEngine {
    /// Base directory for storing memory files
    base_dir: PathBuf,

    /// In-memory cache of recently accessed nodes
    cache: HashMap<Uuid, MemoryNode>,

    /// Maximum number of nodes to keep in the cache
    max_cache_size: usize,
}

impl FileStorageEngine {
    /// Creates a new file storage engine with the specified base directory
    ///
    /// # Arguments
    ///
    /// * `base_dir` - Path to the directory where memory nodes will be stored
    ///
    /// # Returns
    ///
    /// A new FileStorageEngine instance
    pub fn new<P: AsRef<Path>>(base_dir: P) -> Result<Self> {
        let base_dir = base_dir.as_ref().to_path_buf();

        // Create the base directory if it doesn't exist
        if !base_dir.exists() {
            fs::create_dir_all(&base_dir)
                .map_err(|e| EngramDbError::Storage(format!("Failed to create directory: {}", e)))?;
        }

        Ok(Self {
            base_dir,
            cache: HashMap::new(),
            max_cache_size: 100, // Cache up to 100 nodes by default
        })
    }

    /// Gets the file path for a memory node
    fn get_file_path(&self, id: Uuid) -> PathBuf {
        // Use the first two characters of the UUID as a subdirectory
        // to avoid having too many files in a single directory
        let id_str = id.to_string();
        let subdir = &id_str[0..2];

        let subdir_path = self.base_dir.join(subdir);

        // Create the subdirectory if it doesn't exist
        if !subdir_path.exists() {
            let _ = fs::create_dir_all(&subdir_path);
        }

        subdir_path.join(format!("{}.json", id_str))
    }

    /// Serializes a memory node to JSON
    fn serialize(&self, node: &MemoryNode) -> Result<String> {
        serde_json::to_string(node)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to serialize node: {}", e)))
    }

    /// Deserializes a memory node from JSON
    fn deserialize(&self, json: &str) -> Result<MemoryNode> {
        serde_json::from_str(json)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to deserialize node: {}", e)))
    }

    /// Updates the cache with a memory node, removing oldest entries if needed
    fn update_cache(&mut self, node: MemoryNode) {
        let id = node.id();

        // If cache is full and this is a new entry, remove the oldest
        if self.cache.len() >= self.max_cache_size && !self.cache.contains_key(&id) {
            // TODO: Implement a more sophisticated cache eviction policy
            // For now, just remove a random entry
            if let Some(evict_id) = self.cache.keys().next().copied() {
                self.cache.remove(&evict_id);
            }
        }

        self.cache.insert(id, node);
    }

    /// Sets the maximum cache size
    pub fn set_max_cache_size(&mut self, size: usize) {
        self.max_cache_size = size;

        // If the new size is smaller than the current cache size,
        // remove entries until we're under the limit
        while self.cache.len() > self.max_cache_size {
            if let Some(evict_id) = self.cache.keys().next().copied() {
                self.cache.remove(&evict_id);
            } else {
                break;
            }
        }
    }
}

impl StorageEngine for FileStorageEngine {
    fn save(&mut self, node: &MemoryNode) -> Result<()> {
        let json = self.serialize(node)?;
        let file_path = self.get_file_path(node.id());

        // Write to file
        let mut file = File::create(file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to create file: {}", e)))?;

        file.write_all(json.as_bytes())
            .map_err(|e| EngramDbError::Storage(format!("Failed to write file: {}", e)))?;

        // Update cache
        self.update_cache(node.clone());

        Ok(())
    }

    fn load(&self, id: Uuid) -> Result<MemoryNode> {
        // Check cache first
        if let Some(node) = self.cache.get(&id) {
            return Ok(node.clone());
        }

        let file_path = self.get_file_path(id);

        // Check if file exists
        if !file_path.exists() {
            return Err(EngramDbError::Storage(format!(
                "Memory node not found: {}",
                id
            )));
        }

        // Read from file
        let mut file = File::open(file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to open file: {}", e)))?;

        let mut json = String::new();
        file.read_to_string(&mut json)
            .map_err(|e| EngramDbError::Storage(format!("Failed to read file: {}", e)))?;

        // Deserialize and return
        self.deserialize(&json)
    }

    fn delete(&mut self, id: Uuid) -> Result<()> {
        let file_path = self.get_file_path(id);

        // Check if file exists
        if !file_path.exists() {
            return Err(EngramDbError::Storage(format!(
                "Memory node not found: {}",
                id
            )));
        }

        // Remove from cache
        self.cache.remove(&id);

        // Delete file
        fs::remove_file(file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to delete file: {}", e)))?;

        Ok(())
    }

    fn list_all(&self) -> Result<Vec<Uuid>> {
        let mut ids = Vec::new();

        // Walk through the directory structure
        for entry in fs::read_dir(&self.base_dir)
            .map_err(|e| EngramDbError::Storage(format!("Failed to read directory: {}", e)))?
        {
            let entry = entry.map_err(|e| {
                EngramDbError::Storage(format!("Failed to read directory entry: {}", e))
            })?;

            let path = entry.path();

            // Skip non-directories
            if !path.is_dir() {
                continue;
            }

            // Read all JSON files in the subdirectory
            for file_entry in fs::read_dir(&path)
                .map_err(|e| EngramDbError::Storage(format!("Failed to read subdirectory: {}", e)))?
            {
                let file_entry = file_entry.map_err(|e| {
                    EngramDbError::Storage(format!("Failed to read file entry: {}", e))
                })?;

                let file_path = file_entry.path();

                // Skip non-JSON files
                if !file_path.is_file()
                    || file_path.extension().and_then(|ext| ext.to_str()) != Some("json")
                {
                    continue;
                }

                // Extract UUID from filename
                if let Some(file_name) = file_path.file_stem().and_then(|name| name.to_str()) {
                    if let Ok(id) = Uuid::parse_str(file_name) {
                        ids.push(id);
                    }
                }
            }
        }

        Ok(ids)
    }
    
    fn get_type(&self) -> crate::database::StorageType {
        crate::database::StorageType::MultiFile
    }
    
    fn get_path(&self) -> Option<&std::path::Path> {
        Some(&self.base_dir)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_file_storage_crud() {
        // Create a temporary directory for testing
        let temp_dir = tempdir().unwrap();
        let mut storage = FileStorageEngine::new(temp_dir.path()).unwrap();

        // Create a test memory node
        let node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        let id = node.id();

        // Test save
        storage.save(&node).unwrap();

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

        // Verify it's gone
        let result = storage.load(id);
        assert!(result.is_err());

        let ids = storage.list_all().unwrap();
        assert_eq!(ids.len(), 0);
    }
}