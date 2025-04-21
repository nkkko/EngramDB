use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use uuid::Uuid;

use super::StorageEngine;
use crate::core::MemoryNode;
use crate::error::EngramDbError;
use crate::Result;

/// Header information for the database file
#[derive(serde::Serialize, serde::Deserialize)]
struct DatabaseHeader {
    /// Magic bytes to identify the file format
    magic: [u8; 8],
    /// Format version
    version: u32,
    /// Number of nodes stored
    node_count: u64,
    /// Index offset within the file
    index_offset: u64,
}

/// Entry in the index mapping UUIDs to file positions
#[derive(serde::Serialize, serde::Deserialize)]
struct IndexEntry {
    /// UUID of the memory node
    id: Uuid,
    /// Position of the node data within the file
    offset: u64,
    /// Size of the serialized node data
    size: u32,
}

/// A storage engine that keeps all memory nodes in a single file
///
/// This storage engine provides more efficient persistence compared to
/// the FileStorageEngine by storing all nodes in a single structured file.
pub struct SingleFileStorageEngine {
    /// Path to the database file
    file_path: PathBuf,
    /// Memory node index mapping UUIDs to file positions
    index: HashMap<Uuid, IndexEntry>,
    /// In-memory cache of recently accessed nodes
    cache: HashMap<Uuid, MemoryNode>,
    /// Maximum number of nodes to keep in the cache
    max_cache_size: usize,
}

/// Magic bytes for identifying EngramDB files: "ENGRAMDB"
const MAGIC: [u8; 8] = [0x45, 0x4E, 0x47, 0x52, 0x41, 0x4D, 0x44, 0x42];
/// Current file format version
const VERSION: u32 = 1;
/// Size of the header in bytes - ensure it's large enough
const HEADER_SIZE: u64 = 28; // Magic (8) + Version (4) + Count (8) + Offset (8)

impl SingleFileStorageEngine {
    /// Creates a new single-file storage engine with the specified file path
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the database file
    ///
    /// # Returns
    ///
    /// A new SingleFileStorageEngine instance
    pub fn new<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let file_exists = file_path.exists();
        
        // Create parent directory if it doesn't exist
        if let Some(parent) = file_path.parent() {
            if !parent.exists() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| EngramDbError::Storage(format!("Failed to create directory: {}", e)))?;
            }
        }
        
        let mut index = HashMap::new();
        
        if file_exists {
            // Load existing database file
            index = Self::load_index(&file_path)?;
        } else {
            // Create a new database file with header
            let file = OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&file_path)
                .map_err(|e| EngramDbError::Storage(format!("Failed to create database file: {}", e)))?;
                
            // Initialize with empty database
            Self::initialize_file(file)?;
        }
        
        Ok(Self {
            file_path,
            index,
            cache: HashMap::new(),
            max_cache_size: 100, // Cache up to 100 nodes by default
        })
    }
    
    /// Initializes a new database file with header
    fn initialize_file(mut file: File) -> Result<()> {
        // Create empty header
        let header = DatabaseHeader {
            magic: MAGIC,
            version: VERSION,
            node_count: 0,
            index_offset: HEADER_SIZE, // Initially, index starts right after header
        };
        
        // Serialize header - make sure to fill the exact size
        let mut header_bytes = vec![0; HEADER_SIZE as usize];
        let serialized = bincode::serialize(&header)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to serialize header: {}", e)))?;
        
        // Copy serialized data to our fixed-size buffer
        header_bytes[..serialized.len()].copy_from_slice(&serialized);
            
        // Write header
        file.write_all(&header_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to write header: {}", e)))?;
            
        // Write empty index but with a valid size
        // Serialize an empty HashMap for the index
        let empty_index: HashMap<Uuid, IndexEntry> = HashMap::new();
        let index_bytes = bincode::serialize(&empty_index)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to serialize empty index: {}", e)))?;
        
        let index_size = index_bytes.len() as u64;
        file.write_all(&index_size.to_le_bytes())
            .map_err(|e| EngramDbError::Storage(format!("Failed to write index size: {}", e)))?;
        
        // Write the empty index structure
        file.write_all(&index_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to write empty index: {}", e)))?;
            
        Ok(())
    }
    
    /// Loads the index from an existing database file
    fn load_index(file_path: &Path) -> Result<HashMap<Uuid, IndexEntry>> {
        let mut file = OpenOptions::new()
            .read(true)
            .open(file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to open database file: {}", e)))?;
        
        // Get file size to check if it's valid
        let file_size = file.metadata()
            .map_err(|e| EngramDbError::Storage(format!("Failed to get file metadata: {}", e)))?
            .len();
            
        if file_size < HEADER_SIZE + 8 {  // Header + at least 8 bytes for index size
            return Err(EngramDbError::Storage("Database file is too small or corrupt".to_string()));
        }
            
        // Read and validate header
        let mut header_bytes = vec![0; HEADER_SIZE as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to read header: {}", e)))?;
            
        let header: DatabaseHeader = bincode::deserialize(&header_bytes)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to deserialize header: {}", e)))?;
            
        // Validate magic bytes
        if header.magic != MAGIC {
            return Err(EngramDbError::Storage("Invalid database file format".to_string()));
        }
        
        // Seek to index position
        file.seek(SeekFrom::Start(header.index_offset))
            .map_err(|e| EngramDbError::Storage(format!("Failed to seek to index: {}", e)))?;
            
        // Read index size
        let mut size_bytes = [0u8; 8];
        file.read_exact(&mut size_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to read index size: {}", e)))?;
            
        let index_size = u64::from_le_bytes(size_bytes);
        
        // Validate index size against file size
        if header.index_offset + 8 + index_size > file_size {
            return Err(EngramDbError::Storage("Database file is corrupt - index size exceeds file size".to_string()));
        }
        
        // Read empty index
        if index_size == 0 {
            return Ok(HashMap::new());
        }
        
        // Read index data
        let mut index_bytes = vec![0; index_size as usize];
        file.read_exact(&mut index_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to read index: {}", e)))?;
            
        // Deserialize index
        let index: HashMap<Uuid, IndexEntry> = bincode::deserialize(&index_bytes)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to deserialize index: {}", e)))?;
            
        Ok(index)
    }
    
    /// Updates the index in the database file
    fn update_index(&self) -> Result<()> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .open(&self.file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to open database file: {}", e)))?;
            
        // Read current header
        let mut header_bytes = vec![0; HEADER_SIZE as usize];
        file.read_exact(&mut header_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to read header: {}", e)))?;
            
        let mut header: DatabaseHeader = bincode::deserialize(&header_bytes)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to deserialize header: {}", e)))?;
            
        // Serialize the index
        let index_bytes = bincode::serialize(&self.index)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to serialize index: {}", e)))?;
            
        // Update header with new node count
        header.node_count = self.index.len() as u64;
        
        // Write updated header
        let updated_header_bytes = bincode::serialize(&header)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to serialize header: {}", e)))?;
            
        file.seek(SeekFrom::Start(0))
            .map_err(|e| EngramDbError::Storage(format!("Failed to seek to header: {}", e)))?;
            
        file.write_all(&updated_header_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to write header: {}", e)))?;
            
        // Write index size and data
        file.seek(SeekFrom::Start(header.index_offset))
            .map_err(|e| EngramDbError::Storage(format!("Failed to seek to index: {}", e)))?;
            
        let index_size = index_bytes.len() as u64;
        file.write_all(&index_size.to_le_bytes())
            .map_err(|e| EngramDbError::Storage(format!("Failed to write index size: {}", e)))?;
            
        file.write_all(&index_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to write index: {}", e)))?;
            
        Ok(())
    }
    
    /// Updates the cache with a memory node, removing oldest entries if needed
    fn update_cache(&mut self, node: MemoryNode) {
        let id = node.id();
        
        // If cache is full and this is a new entry, remove the oldest
        if self.cache.len() >= self.max_cache_size && !self.cache.contains_key(&id) {
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
    
    /// Finds the position to write a node in the file
    fn find_write_position(&self, node_id: Uuid, data_size: usize) -> Result<u64> {
        // Determine position based on whether we're updating or adding
        if let Some(entry) = self.index.get(&node_id) {
            // Updating existing node
            if entry.size as usize >= data_size {
                // We can reuse the existing position
                return Ok(entry.offset);
            }
        }
        
        // For new nodes or nodes that need more space, append to the end of the file
        let file = OpenOptions::new()
            .read(true)
            .open(&self.file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to open database file: {}", e)))?;
        
        // Get current file size and append to it
        let file_size = file.metadata()
            .map_err(|e| EngramDbError::Storage(format!("Failed to get file size: {}", e)))?
            .len();
            
        Ok(file_size)
    }
    
    /// Compacts the database file by removing deleted nodes and fragmentation
    pub fn compact(&mut self) -> Result<()> {
        // Create a temporary file
        let temp_path = self.file_path.with_file_name(format!(
            "{}.temp",
            self.file_path.file_name().unwrap().to_string_lossy()
        ));
        
        // Initialize the temporary file
        let temp_file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&temp_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to create temporary file: {}", e)))?;
            
        Self::initialize_file(temp_file)?;
        
        // Create a new engine for the temporary file
        let mut temp_engine = SingleFileStorageEngine::new(&temp_path)?;
        
        // Copy all nodes to the temporary file
        for id in self.index.keys().copied().collect::<Vec<_>>() {
            if let Ok(node) = self.load(id) {
                temp_engine.save(&node)?;
            }
        }
        
        // Replace the original file with the temporary file
        std::fs::rename(&temp_path, &self.file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to replace database file: {}", e)))?;
            
        // Update our state
        self.index = temp_engine.index;
        
        Ok(())
    }
}

impl StorageEngine for SingleFileStorageEngine {
    fn save(&mut self, node: &MemoryNode) -> Result<()> {
        // Serialize the node
        let node_bytes = bincode::serialize(node)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to serialize node: {}", e)))?;
            
        // Find position to write
        let position = self.find_write_position(node.id(), node_bytes.len())?;
        
        // Open file for writing - make sure it exists first
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)  // Create if it doesn't exist
            .open(&self.file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to open database file: {}", e)))?;
        
        // If this is a new file, initialize it
        if file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            Self::initialize_file(file)?;
            
            // Reopen the file
            file = OpenOptions::new()
                .read(true)
                .write(true)
                .open(&self.file_path)
                .map_err(|e| EngramDbError::Storage(format!("Failed to reopen database file: {}", e)))?;
        }
            
        // Write node data
        file.seek(SeekFrom::Start(position))
            .map_err(|e| EngramDbError::Storage(format!("Failed to seek to position: {}", e)))?;
            
        file.write_all(&node_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to write node data: {}", e)))?;
            
        // Update index
        self.index.insert(node.id(), IndexEntry {
            id: node.id(),
            offset: position,
            size: node_bytes.len() as u32,
        });
        
        // Update the index in the file
        self.update_index()?;
        
        // Update cache
        self.update_cache(node.clone());
        
        Ok(())
    }
    
    fn load(&self, id: Uuid) -> Result<MemoryNode> {
        // Check cache first
        if let Some(node) = self.cache.get(&id) {
            return Ok(node.clone());
        }
        
        // Check if node exists in index
        let entry = self.index.get(&id).ok_or_else(|| {
            EngramDbError::Storage(format!("Memory node not found: {}", id))
        })?;
        
        // Open file for reading
        let mut file = OpenOptions::new()
            .read(true)
            .open(&self.file_path)
            .map_err(|e| EngramDbError::Storage(format!("Failed to open database file: {}", e)))?;
            
        // Seek to node position
        file.seek(SeekFrom::Start(entry.offset))
            .map_err(|e| EngramDbError::Storage(format!("Failed to seek to node: {}", e)))?;
            
        // Read node data
        let mut node_bytes = vec![0; entry.size as usize];
        file.read_exact(&mut node_bytes)
            .map_err(|e| EngramDbError::Storage(format!("Failed to read node data: {}", e)))?;
            
        // Deserialize node
        let node: MemoryNode = bincode::deserialize(&node_bytes)
            .map_err(|e| EngramDbError::Serialization(format!("Failed to deserialize node: {}", e)))?;
            
        // Since we can't update the cache here (self is immutable), 
        // we return the node and let the caller update the cache if needed
        Ok(node)
    }
    
    fn delete(&mut self, id: Uuid) -> Result<()> {
        // Check if node exists
        if !self.index.contains_key(&id) {
            return Err(EngramDbError::Storage(format!(
                "Memory node not found: {}",
                id
            )));
        }
        
        // Remove from index
        self.index.remove(&id);
        
        // Remove from cache
        self.cache.remove(&id);
        
        // Update the index in the file
        self.update_index()?;
        
        Ok(())
    }
    
    fn list_all(&self) -> Result<Vec<Uuid>> {
        Ok(self.index.keys().copied().collect())
    }
    
    fn get_type(&self) -> crate::database::StorageType {
        crate::database::StorageType::SingleFile
    }
    
    fn get_path(&self) -> Option<&std::path::Path> {
        Some(&self.file_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    
    #[test]
    fn test_single_file_storage_crud() {
        // Create a temporary directory for testing
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test.engramdb");
        
        let mut storage = SingleFileStorageEngine::new(&db_path).unwrap();
        
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
    
    #[test]
    fn test_single_file_storage_multiple_nodes() {
        // Create a temporary directory for testing
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_multi.engramdb");
        
        let mut storage = SingleFileStorageEngine::new(&db_path).unwrap();
        
        // Create and save 5 nodes
        let mut ids = Vec::new();
        for i in 0..5 {
            let node = MemoryNode::new(vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3]);
            let id = node.id();
            storage.save(&node).unwrap();
            ids.push(id);
        }
        
        // Verify we can load all nodes
        for id in &ids {
            let node = storage.load(*id).unwrap();
            assert_eq!(node.id(), *id);
        }
        
        // Verify list_all returns all IDs
        let all_ids = storage.list_all().unwrap();
        assert_eq!(all_ids.len(), 5);
        for id in ids {
            assert!(all_ids.contains(&id));
        }
    }
    
    #[test]
    fn test_single_file_storage_persistence() {
        // Create a temporary directory for testing
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_persist.engramdb");
        
        // Create initial storage and save a node
        let node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        let id = node.id();
        
        {
            let mut storage = SingleFileStorageEngine::new(&db_path).unwrap();
            storage.save(&node).unwrap();
        }
        
        // Create a new storage instance and verify the node is still there
        {
            let storage = SingleFileStorageEngine::new(&db_path).unwrap();
            let loaded_node = storage.load(id).unwrap();
            assert_eq!(loaded_node.id(), id);
            assert_eq!(loaded_node.embeddings(), node.embeddings());
        }
    }
    
    #[test]
    fn test_single_file_storage_compact() {
        // Create a temporary directory for testing
        let temp_dir = tempdir().unwrap();
        let db_path = temp_dir.path().join("test_compact.engramdb");
        
        let mut storage = SingleFileStorageEngine::new(&db_path).unwrap();
        
        // Create and save 10 nodes
        let mut ids = Vec::new();
        for i in 0..10 {
            let node = MemoryNode::new(vec![i as f32 * 0.1, i as f32 * 0.2, i as f32 * 0.3]);
            let id = node.id();
            storage.save(&node).unwrap();
            ids.push(id);
        }
        
        // Delete half of them
        for i in 0..5 {
            storage.delete(ids[i]).unwrap();
        }
        
        // Compact the database
        storage.compact().unwrap();
        
        // Verify the remaining nodes are still there
        for i in 5..10 {
            let node = storage.load(ids[i]).unwrap();
            assert_eq!(node.id(), ids[i]);
        }
        
        // Verify the deleted nodes are gone
        for i in 0..5 {
            let result = storage.load(ids[i]);
            assert!(result.is_err());
        }
    }
}