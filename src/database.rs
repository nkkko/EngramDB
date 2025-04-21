//! The EngramDB Database
//!
//! This module provides a unified interface to the EngramDB database system,
//! combining storage, vector search, and query capabilities.

use crate::core::{Connection, MemoryNode};
use crate::RelationshipType;
use crate::query::QueryBuilder;
use crate::storage::{FileStorageEngine, MemoryStorageEngine, SingleFileStorageEngine, StorageEngine};
use crate::vector::{VectorIndex, VectorSearchIndex, VectorIndexConfig, VectorAlgorithm, HnswConfig, create_vector_index};
use crate::error::EngramDbError;
use crate::Result;

use std::path::Path;
use uuid::Uuid;
use serde::{Serialize, Deserialize};

/// Represents connection information returned by the database
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionInfo {
    /// ID of the target memory node
    pub target_id: Uuid,
    
    /// Type of relationship as a string
    pub type_name: String,
    
    /// Strength of the connection (0.0 to 1.0)
    pub strength: f32,
}

/// Storage type options for the database
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StorageType {
    /// In-memory storage (volatile)
    Memory,
    
    /// Multi-file storage (one file per node)
    MultiFile,
    
    /// Single-file storage (all nodes in one file)
    SingleFile,
}

/// Configuration options for the EngramDB database
#[derive(Debug, Clone)]
pub struct DatabaseConfig {
    /// The type of storage engine to use
    pub storage_type: StorageType,
    
    /// Path for storage (directory for MultiFile, file for SingleFile)
    pub storage_path: Option<String>,
    
    /// Size of the query result cache (0 to disable caching)
    pub cache_size: usize,
    
    /// Vector index configuration
    pub vector_index_config: VectorIndexConfig,
}

impl Default for DatabaseConfig {
    fn default() -> Self {
        Self {
            storage_type: StorageType::MultiFile,
            storage_path: Some("./engramdb_storage".to_string()),
            cache_size: 100,
            vector_index_config: VectorIndexConfig::default(),
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
    pub vector_index: Box<dyn VectorSearchIndex + Send + Sync>,
}

// Intentionally not implementing Clone for Database because it's not safely clonable.
// For thread-safe usage that shares the same database, use ThreadSafeDatabase instead.
// For Python binding support, we provide methods to convert to thread-safe versions.

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
        let storage: Box<dyn StorageEngine> = match config.storage_type {
            StorageType::Memory => Box::new(MemoryStorageEngine::new()),
            
            StorageType::MultiFile | StorageType::SingleFile => {
                let path = match &config.storage_path {
                    Some(path) => path,
                    None => return Err(EngramDbError::Other("Storage path must be specified for file-based storage".to_string())),
                };
                
                match config.storage_type {
                    StorageType::MultiFile => Box::new(FileStorageEngine::new(Path::new(path))?),
                    StorageType::SingleFile => Box::new(SingleFileStorageEngine::new(Path::new(path))?),
                    _ => unreachable!(),
                }
            }
        };
        
        let vector_index = create_vector_index(&config.vector_index_config);
        
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
            vector_index: Box::new(VectorIndex::new()),
        }
    }
    
    /// Creates a new in-memory database with HNSW index for faster vector search
    pub fn in_memory_with_hnsw() -> Self {
        let config = DatabaseConfig {
            storage_type: StorageType::Memory,
            storage_path: None,
            cache_size: 100,
            vector_index_config: VectorIndexConfig {
                algorithm: VectorAlgorithm::HNSW,
                hnsw: Some(HnswConfig::default()),
                multi_vector: None,
            },
        };
        
        Self {
            storage: Box::new(MemoryStorageEngine::new()),
            vector_index: create_vector_index(&config.vector_index_config),
        }
    }
    
    /// Creates a file-based database with the specified storage path and HNSW index
    pub fn file_based_with_hnsw<P: AsRef<Path>>(path: P) -> Result<Self> {
        let config = DatabaseConfig {
            storage_type: StorageType::MultiFile,
            storage_path: Some(path.as_ref().to_string_lossy().to_string()),
            cache_size: 100,
            vector_index_config: VectorIndexConfig {
                algorithm: VectorAlgorithm::HNSW,
                hnsw: Some(HnswConfig::default()),
                multi_vector: None,
            },
        };
        
        Self::new(config)
    }
    
    /// Creates a multi-file based database at the specified directory
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
        
        let mut db = Self {
            storage: Box::new(storage),
            vector_index: Box::new(VectorIndex::new()),
        };
        
        // Initialize by loading existing memories
        let _ = db.initialize();
        
        Ok(db)
    }
    
    /// Creates a single-file based database at the specified file path
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the database file
    ///
    /// # Returns
    ///
    /// A new Database instance, or an error if initialization failed
    pub fn single_file<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        let file_path = file_path.as_ref().to_path_buf();
        let storage = SingleFileStorageEngine::new(&file_path)?;
        
        let mut db = Self {
            storage: Box::new(storage),
            vector_index: Box::new(VectorIndex::new()),
        };
        
        // Initialize by loading existing memories
        let _ = db.initialize();
        
        Ok(db)
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
    /// * `connected_to` - Optional ID to filter results by connection
    /// * `relationship_type` - Optional relationship type for connection filtering
    ///
    /// # Returns
    ///
    /// A vector of (UUID, similarity) pairs, sorted by descending similarity
    pub fn search_similar(
        &self, 
        query_vector: &[f32], 
        limit: usize, 
        threshold: f32,
        connected_to: Option<Uuid>,
        relationship_type: Option<String>,
    ) -> Result<Vec<(Uuid, f32)>> {
        // Perform the initial vector search
        let mut results = self.vector_index.search(query_vector, limit * 2, threshold)?;
        
        // If no connection filtering is required, just return the results
        if connected_to.is_none() {
            // Limit to the requested number of results
            results.truncate(limit);
            return Ok(results);
        }
        
        // If we need to filter by connections
        let source_id = connected_to.unwrap();
        
        // Try to load the source memory
        let source_memory = match self.load(source_id) {
            Ok(memory) => memory,
            Err(e) => {
                return Err(EngramDbError::Other(format!("Failed to load source memory for connection filtering: {}", e)));
            }
        };
        
        // Get all connections from the source memory
        let connections = source_memory.connections();
        
        // Filter results to only include connected memories
        let filtered_results: Vec<(Uuid, f32)> = results
            .into_iter()
            .filter(|(memory_id, _)| {
                // Check if this memory is connected to the source
                connections.iter().any(|conn| {
                    // First check if the target ID matches
                    if conn.target_id() != *memory_id {
                        return false;
                    }
                    
                    // Then check if relationship type matches, if specified
                    if let Some(rel_type) = &relationship_type {
                        let matches = match conn.relationship_type() {
                            RelationshipType::Association => rel_type == "Association",
                            RelationshipType::Causation => rel_type == "Causation",
                            RelationshipType::PartOf => rel_type == "PartOf",
                            RelationshipType::Contains => rel_type == "Contains",
                            RelationshipType::Sequence => rel_type == "Sequence",
                            RelationshipType::Custom(custom_type) => rel_type == custom_type,
                        };
                        matches
                    } else {
                        // If no relationship type specified, any connection is fine
                        true
                    }
                })
            })
            .take(limit)
            .collect();
        
        Ok(filtered_results)
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
    
    /// Creates a memory node from text and saves it to the database
    ///
    /// # Arguments
    ///
    /// * `text` - The text content to create a memory from
    /// * `embedding_service` - The embedding service to use
    /// * `category` - Optional category for the content
    /// * `attributes` - Optional additional attributes to set
    ///
    /// # Returns
    ///
    /// The ID of the newly created memory node
    #[cfg(feature = "embeddings")]
    pub fn create_memory_from_text(
        &mut self,
        text: &str,
        embedding_service: &crate::embeddings::EmbeddingService,
        category: Option<&str>,
        attributes: Option<&std::collections::HashMap<String, crate::core::AttributeValue>>,
    ) -> Result<uuid::Uuid> {
        // Create the memory node from text
        let mut memory = match crate::core::MemoryNode::from_text(text, embedding_service, category) {
            Ok(node) => node,
            Err(e) => return Err(crate::error::EngramDbError::Other(format!("Failed to create memory from text: {}", e))),
        };
        
        // Add any additional attributes
        if let Some(attrs) = attributes {
            for (key, value) in attrs {
                memory.set_attribute(key.clone(), value.clone());
            }
        }
        
        // Save to database
        self.save(&memory)
    }
    
    /// Creates a connection between two memory nodes
    ///
    /// # Arguments
    ///
    /// * `source_id` - The UUID of the source memory node
    /// * `target_id` - The UUID of the target memory node
    /// * `relationship_type` - The type of relationship as a string
    /// * `strength` - The strength of the connection (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// `Ok(())` if the connection was created successfully
    pub fn connect(&mut self, source_id: Uuid, target_id: Uuid, relationship_type: String, strength: f32) -> Result<()> {
        // First, verify that both memories exist
        let _target_memory = self.load(target_id)?;
        
        // Load the source memory
        let mut source_memory = self.load(source_id)?;
        
        // Convert string relationship type to enum
        let rel_type = match relationship_type.as_str() {
            "Association" => RelationshipType::Association,
            "Causation" => RelationshipType::Causation,
            "PartOf" => RelationshipType::PartOf,
            "Contains" => RelationshipType::Contains,
            "Sequence" => RelationshipType::Sequence,
            _ => RelationshipType::Custom(relationship_type),
        };
        
        // Create the connection
        let connection = Connection::new(target_id, rel_type, strength);
        
        // Add the connection to the source memory
        source_memory.add_connection(connection);
        
        // Save the updated source memory
        self.save(&source_memory)?;
        
        Ok(())
    }
    
    /// Removes a connection between two memory nodes
    ///
    /// # Arguments
    ///
    /// * `source_id` - The UUID of the source memory node
    /// * `target_id` - The UUID of the target memory node
    ///
    /// # Returns
    ///
    /// `Ok(true)` if the connection was found and removed, `Ok(false)` if the connection doesn't exist
    pub fn disconnect(&mut self, source_id: Uuid, target_id: Uuid) -> Result<bool> {
        // Load the source memory
        let mut source_memory = self.load(source_id)?;
        
        // Get current connections
        let connections = source_memory.connections().to_vec();
        
        // Find the index of the connection to remove
        let connection_idx = connections.iter().position(|c| c.target_id() == target_id);
        
        if let Some(idx) = connection_idx {
            // Remove the connection 
            let mut updated_connections = connections.clone();
            updated_connections.remove(idx);
            
            // Replace connections in the memory node
            // Note: This is not ideal - we should add a remove_connection method to MemoryNode
            source_memory = self.load(source_id)?;  // Reload to avoid losing other changes
            source_memory.set_attribute("_tmp_connections".to_string(), 
                crate::core::AttributeValue::String("placeholder".to_string()));
            
            // Now add all connections except the one we want to remove
            for conn in connections.iter() {
                if conn.target_id() != target_id {
                    source_memory.add_connection(conn.clone());
                }
            }
            
            // Save the updated source memory
            self.save(&source_memory)?;
            
            Ok(true)
        } else {
            Ok(false)
        }
    }
    
    /// Gets all connections from a specific memory
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The UUID of the memory node
    /// * `relationship_type` - Optional filter for relationship type
    ///
    /// # Returns
    ///
    /// A vector of connection information objects
    pub fn get_connections(&self, memory_id: Uuid, relationship_type: Option<String>) -> Result<Vec<ConnectionInfo>> {
        // Load the memory
        let memory = self.load(memory_id)?;
        
        // Filter connections by relationship type if specified
        let connections: Vec<ConnectionInfo> = memory.connections()
            .iter()
            .filter(|conn| {
                if let Some(rel_type) = &relationship_type {
                    match conn.relationship_type() {
                        RelationshipType::Association => rel_type == "Association",
                        RelationshipType::Causation => rel_type == "Causation",
                        RelationshipType::PartOf => rel_type == "PartOf",
                        RelationshipType::Contains => rel_type == "Contains",
                        RelationshipType::Sequence => rel_type == "Sequence",
                        RelationshipType::Custom(custom_type) => rel_type == custom_type,
                    }
                } else {
                    true
                }
            })
            .map(|conn| {
                let type_str = match conn.relationship_type() {
                    RelationshipType::Association => "Association".to_string(),
                    RelationshipType::Causation => "Causation".to_string(),
                    RelationshipType::PartOf => "PartOf".to_string(),
                    RelationshipType::Contains => "Contains".to_string(),
                    RelationshipType::Sequence => "Sequence".to_string(),
                    RelationshipType::Custom(custom_type) => custom_type.clone(),
                };
                
                ConnectionInfo {
                    target_id: conn.target_id(),
                    type_name: type_str,
                    strength: conn.strength(),
                }
            })
            .collect();
        
        Ok(connections)
    }
    
    /// Gets all memories that connect to this memory
    ///
    /// # Arguments
    ///
    /// * `memory_id` - The UUID of the memory node
    /// * `relationship_type` - Optional filter for relationship type
    ///
    /// # Returns
    ///
    /// A vector of connection information objects
    pub fn get_connected_to(&self, memory_id: Uuid, relationship_type: Option<String>) -> Result<Vec<ConnectionInfo>> {
        // Get all memories
        let memory_ids = self.list_all()?;
        
        // Check each memory for connections to this memory
        let mut incoming_connections = Vec::new();
        
        for id in memory_ids {
            if id == memory_id {
                continue;  // Skip the target memory itself
            }
            
            // Load the memory
            if let Ok(memory) = self.load(id) {
                // Find connections to the target memory
                for conn in memory.connections() {
                    if conn.target_id() == memory_id {
                        // Check if it matches the relationship type filter
                        let matches_filter = if let Some(rel_type) = &relationship_type {
                            match conn.relationship_type() {
                                RelationshipType::Association => rel_type == "Association",
                                RelationshipType::Causation => rel_type == "Causation",
                                RelationshipType::PartOf => rel_type == "PartOf",
                                RelationshipType::Contains => rel_type == "Contains",
                                RelationshipType::Sequence => rel_type == "Sequence",
                                RelationshipType::Custom(custom_type) => rel_type == custom_type,
                            }
                        } else {
                            true
                        };
                        
                        if matches_filter {
                            let type_str = match conn.relationship_type() {
                                RelationshipType::Association => "Association".to_string(),
                                RelationshipType::Causation => "Causation".to_string(),
                                RelationshipType::PartOf => "PartOf".to_string(),
                                RelationshipType::Contains => "Contains".to_string(),
                                RelationshipType::Sequence => "Sequence".to_string(),
                                RelationshipType::Custom(custom_type) => custom_type.clone(),
                            };
                            
                            incoming_connections.push(ConnectionInfo {
                                target_id: id,  // This is actually the source in this context
                                type_name: type_str,
                                strength: conn.strength(),
                            });
                        }
                    }
                }
            }
        }
        
        Ok(incoming_connections)
    }
    
    /// Clears all memories and connections from the database
    pub fn clear_all(&mut self) -> Result<()> {
        // Get all memory IDs
        let memory_ids = self.list_all()?;
        
        // Delete each memory
        for id in memory_ids {
            let _ = self.delete(id);
        }
        
        // Reset the vector index
        self.vector_index = create_vector_index(&VectorIndexConfig::default());
        
        Ok(())
    }
    
    /// Converts this database into a thread-safe database
    ///
    /// This method converts a standard Database instance into a ThreadSafeDatabase,
    /// which can be safely shared between threads using Arc and RwLock.
    ///
    /// # Returns
    ///
    /// A new thread-safe database instance with the same data
    pub fn to_thread_safe(self) -> crate::vector::ThreadSafeDatabase {
        crate::vector::ThreadSafeDatabase::from_database(self)
    }
    
    /// Creates a thread-safe database that shares the same storage but with separate 
    /// vector indices
    ///
    /// This is useful for scenarios where you want to share the same underlying storage
    /// but need thread-safe access from multiple threads with potentially different
    /// vector index configurations.
    ///
    /// # Arguments
    ///
    /// * `config` - Optional vector index configuration for the new instance
    ///
    /// # Returns
    ///
    /// A result containing either the new thread-safe database or an error
    pub fn create_thread_safe_view(&self, config: Option<VectorIndexConfig>) -> Result<crate::vector::ThreadSafeDatabase> {
        // Create a new database with the same storage type and path
        let storage_path = match self.storage.get_path() {
            Some(path) => Some(path.to_string_lossy().to_string()),
            None => None,
        };
        
        let storage_type = self.storage.get_type();
        
        let vector_config = config.unwrap_or_else(|| VectorIndexConfig::default());
        
        let db_config = DatabaseConfig {
            storage_type,
            storage_path,
            cache_size: 100, // Default cache size
            vector_index_config: vector_config,
        };
        
        // Create and initialize the new database
        let mut db = Self::new(db_config)?;
        db.initialize()?;
        
        // Convert to thread-safe and return
        Ok(db.to_thread_safe())
    }
}

/// A query builder for the database
///
/// This struct wraps the QueryBuilder with a specific database instance,
/// allowing for easier querying.
pub struct DatabaseQueryBuilder<'a> {
    vector_index: &'a Box<dyn VectorSearchIndex + Send + Sync>,
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
    vector_index: &'a Box<dyn VectorSearchIndex + Send + Sync>,
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
        self.builder.execute(self.vector_index.as_ref(), |id| db.load(id))
    }
}