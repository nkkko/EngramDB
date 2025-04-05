//! The EngramDB Database
//!
//! This module provides a unified interface to the EngramDB database system,
//! combining storage, vector search, and query capabilities.

use crate::core::MemoryNode;
use crate::query::QueryBuilder;
use crate::storage::{FileStorageEngine, MemoryStorageEngine, StorageEngine};
use crate::vector::VectorIndex;
use crate::error::EngramDbError;
use crate::Result;

use std::path::Path;
use uuid::Uuid;

/// Configuration options for the EngramDB database
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// Whether to use in-memory storage (volatile) or file-based storage (persistent)
    pub use_memory_storage: bool,
    
    /// Directory path for file storage (ignored if using memory storage)
    pub storage_dir: Option<String>,
    
    /// Size of the query result cache (0 to disable caching)
    pub cache_size: usize,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            use_memory_storage: false,
            storage_dir: Some("./engramdb_storage".to_string()),
            cache_size: 100,
        }
    }
}

/// The EngramDB Database
///
/// This struct provides a unified interface to the memory storage system,
/// vector index, and query engine. It's the main entry point for applications
/// using the EngramDB system.
pub struct Database {
    /// The storage backend (file-based or in-memory)
    storage: Box<dyn StorageEngine>,
    
    /// Vector index for similarity search
    pub vector_index: VectorIndex,
}

// Manual implementation of Clone for Database 
// (needed for Python bindings) using Arc for the trait object
impl Clone for Database {
    fn clone(&self) -> Self {
        // Note: This is a simplified clone that creates a new instance
        // sharing the same vector index but with separate storage.
        // For actual clones, consider using Arc for the storage backend.
        
        // In memory mode, storage is empty after clone
        if self.vector_index.is_empty() {
            return Self::in_memory();
        }
        
        // Create a new in-memory database
        let mut db = Self::in_memory();
        
        // Clone the vector index
        db.vector_index = self.vector_index.clone();
        
        db
    }
}

impl Database {
    /// Creates a new database with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration options for the database
    ///
    /// # Returns
    ///
    /// A new Database instance, or an error if initialization failed
    pub fn new(config: DatabaseConfig) -> Result<Self> {
        let storage: Box<dyn StorageEngine> = if config.use_memory_storage {
            Box::new(MemoryStorageEngine::new())
        } else {
            let path = match &config.storage_dir {
                Some(dir) => dir,
                None => return Err(EngramDbError::Other("Storage directory must be specified for file storage".to_string())),
            };
            
            Box::new(FileStorageEngine::new(Path::new(path))?)
        };
        
        let vector_index = VectorIndex::new();
        
        Ok(Self {
            storage,
            vector_index,
        })
    }
    
    /// Creates a new database with default configuration
    pub fn default() -> Result<Self> {
        Self::new(DatabaseConfig::default())
    }
    
    /// Creates a new in-memory database
    pub fn in_memory() -> Self {
        Self {
            storage: Box::new(MemoryStorageEngine::new()),
            vector_index: VectorIndex::new(),
        }
    }
    
    /// Creates a file-based database at the specified directory
    ///
    /// # Arguments
    ///
    /// * `dir` - Path to the storage directory
    ///
    /// # Returns
    ///
    /// A new Database instance, or an error if initialization failed
    pub fn file_based<P: AsRef<Path>>(dir: P) -> Result<Self> {
        let dir_path = dir.as_ref().to_path_buf();
        let storage = FileStorageEngine::new(&dir_path)?;
        
        Ok(Self {
            storage: Box::new(storage),
            vector_index: VectorIndex::new(),
        })
    }
    
    /// Initializes the database by loading existing memories into the vector index
    pub fn initialize(&mut self) -> Result<()> {
        // Load all existing memories into the vector index
        let memory_ids = self.storage.list_all()?;
        
        for id in memory_ids {
            let node = self.storage.load(id)?;
            self.vector_index.add(&node)?;
        }
        
        Ok(())
    }
    
    /// Saves a memory node to the database
    ///
    /// # Arguments
    ///
    /// * `node` - The memory node to save
    ///
    /// # Returns
    ///
    /// The ID of the saved memory node
    pub fn save(&mut self, node: &MemoryNode) -> Result<Uuid> {
        // Save to storage
        self.storage.save(node)?;
        
        // Add to vector index
        self.vector_index.add(node)?;
        
        Ok(node.id())
    }
    
    /// Loads a memory node by its ID
    ///
    /// # Arguments
    ///
    /// * `id` - The UUID of the memory node to load
    ///
    /// # Returns
    ///
    /// The requested memory node if found
    pub fn load(&self, id: Uuid) -> Result<MemoryNode> {
        self.storage.load(id)
    }
    
    /// Deletes a memory node by its ID
    ///
    /// # Arguments
    ///
    /// * `id` - The UUID of the memory node to delete
    pub fn delete(&mut self, id: Uuid) -> Result<()> {
        // Remove from vector index
        self.vector_index.remove(id)?;
        
        // Remove from storage
        self.storage.delete(id)
    }
    
    /// Gets all memory node IDs in the database
    pub fn list_all(&self) -> Result<Vec<Uuid>> {
        self.storage.list_all()
    }
    
    /// Searches for similar memories using vector similarity
    ///
    /// # Arguments
    ///
    /// * `query_vector` - The vector to compare against
    /// * `limit` - Maximum number of results to return
    /// * `threshold` - Minimum similarity threshold (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// A vector of (UUID, similarity) pairs, sorted by descending similarity
    pub fn search_similar(&self, query_vector: &[f32], limit: usize, threshold: f32) -> Result<Vec<(Uuid, f32)>> {
        self.vector_index.search(query_vector, limit, threshold)
    }
    
    /// Creates a query builder for this database
    pub fn query(&self) -> DatabaseQueryBuilder {
        DatabaseQueryBuilder {
            vector_index: &self.vector_index,
            database: self,
        }
    }
    
    /// Returns the number of memories in the database
    pub fn len(&self) -> Result<usize> {
        Ok(self.list_all()?.len())
    }
    
    /// Checks if the database is empty
    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.list_all()?.is_empty())
    }
}

/// A query builder for the database
///
/// This struct wraps the QueryBuilder with a specific database instance,
/// allowing for easier querying.
pub struct DatabaseQueryBuilder<'a> {
    vector_index: &'a VectorIndex,
    database: &'a Database,
}

impl<'a> DatabaseQueryBuilder<'a> {
    /// Sets the query vector for similarity search
    pub fn with_vector(self, vector: Vec<f32>) -> DatabaseQuery<'a> {
        DatabaseQuery {
            builder: QueryBuilder::new().with_vector(vector),
            vector_index: self.vector_index,
            database: self.database,
        }
    }
    
    /// Restricts the query to only consider the specified IDs
    pub fn with_ids(self, ids: Vec<Uuid>) -> DatabaseQuery<'a> {
        DatabaseQuery {
            builder: QueryBuilder::new().with_include_ids(ids),
            vector_index: self.vector_index,
            database: self.database,
        }
    }
    
    /// Creates an empty query
    pub fn empty(self) -> DatabaseQuery<'a> {
        DatabaseQuery {
            builder: QueryBuilder::new(),
            vector_index: self.vector_index,
            database: self.database,
        }
    }
}

/// A database query with builder methods for adding constraints
pub struct DatabaseQuery<'a> {
    builder: QueryBuilder,
    vector_index: &'a VectorIndex,
    database: &'a Database,
}

impl<'a> DatabaseQuery<'a> {
    /// Adds an attribute filter to the query
    pub fn with_attribute_filter(mut self, filter: crate::query::AttributeFilter) -> Self {
        self.builder = self.builder.with_attribute_filter(filter);
        self
    }
    
    /// Adds a temporal filter to the query
    pub fn with_temporal_filter(mut self, filter: crate::query::TemporalFilter) -> Self {
        self.builder = self.builder.with_temporal_filter(filter);
        self
    }
    
    /// Sets the similarity threshold for vector queries
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.builder = self.builder.with_similarity_threshold(threshold);
        self
    }
    
    /// Sets the maximum number of results to return
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.builder = self.builder.with_limit(limit);
        self
    }
    
    /// Adds IDs to exclude from the results
    pub fn with_exclude_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.builder = self.builder.with_exclude_ids(ids);
        self
    }
    
    /// Executes the query and returns the matching memory nodes
    pub fn execute(self) -> Result<Vec<MemoryNode>> {
        let db = self.database;
        self.builder.execute(self.vector_index, |id| db.load(id))
    }
}