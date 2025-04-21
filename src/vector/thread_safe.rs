//! Thread-safe implementations for EngramDB
//!
//! This module provides thread-safe wrappers for the Database and related components.

use std::sync::{Arc, RwLock, Mutex};
use crate::core::MemoryNode;
use crate::database::{Database, DatabaseConfig};
use crate::error::EngramDbError;
use crate::Result;
use std::path::Path;
use uuid::Uuid;
use crate::query::QueryBuilder;

/// A thread-safe wrapper for the EngramDB Database
///
/// This struct wraps the Database with synchronization primitives to make it
/// safe to share between threads. It provides the same functionality as the Database,
/// but with proper thread safety through Arc and RwLock.
pub struct ThreadSafeDatabase {
    /// The inner database protected by a RwLock for concurrent access
    inner: Arc<RwLock<Database>>,
}

impl ThreadSafeDatabase {
    /// Creates a new thread-safe database with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - Configuration options for the database
    ///
    /// # Returns
    ///
    /// A new thread-safe Database instance, or an error if initialization failed
    pub fn new(config: DatabaseConfig) -> Result<Self> {
        match Database::new(config) {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
        }
    }
    
    /// Creates a new thread-safe database with default configuration
    ///
    /// # Returns
    ///
    /// A new thread-safe Database instance with default configuration
    pub fn default() -> Result<Self> {
        match Database::default() {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
        }
    }
    
    /// Creates a new thread-safe in-memory database
    ///
    /// # Returns
    ///
    /// A new thread-safe in-memory Database instance
    pub fn in_memory() -> Self {
        let db = Database::in_memory();
        Self {
            inner: Arc::new(RwLock::new(db)),
        }
    }
    
    /// Creates a new thread-safe in-memory database with HNSW index
    ///
    /// # Returns
    ///
    /// A new thread-safe in-memory Database instance with HNSW index
    pub fn in_memory_with_hnsw() -> Self {
        let db = Database::in_memory_with_hnsw();
        Self {
            inner: Arc::new(RwLock::new(db)),
        }
    }
    
    /// Creates a new thread-safe file-based database
    ///
    /// # Arguments
    ///
    /// * `dir` - Path to the storage directory
    ///
    /// # Returns
    ///
    /// A new thread-safe file-based Database instance
    pub fn file_based<P: AsRef<Path>>(dir: P) -> Result<Self> {
        match Database::file_based(dir) {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
        }
    }
    
    /// Creates a new thread-safe single-file database
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the database file
    ///
    /// # Returns
    ///
    /// A new thread-safe single-file Database instance
    pub fn single_file<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        match Database::single_file(file_path) {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
        }
    }
    
    /// Creates a thread-safe file-based database with HNSW index
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the storage directory
    ///
    /// # Returns
    ///
    /// A new thread-safe file-based Database instance with HNSW index
    pub fn file_based_with_hnsw<P: AsRef<Path>>(path: P) -> Result<Self> {
        match Database::file_based_with_hnsw(path) {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
        }
    }
    
    /// Creates a new thread-safe database from an existing Database instance
    ///
    /// This is useful for converting an existing Database into a thread-safe version.
    ///
    /// # Arguments
    ///
    /// * `database` - An existing Database instance
    ///
    /// # Returns
    ///
    /// A new thread-safe Database instance wrapping the provided Database
    pub fn from_database(database: Database) -> Self {
        Self {
            inner: Arc::new(RwLock::new(database)),
        }
    }

    /// Saves a memory node to the database
    pub fn save(&self, node: &MemoryNode) -> Result<Uuid> {
        match self.inner.write() {
            Ok(mut db) => db.save(node),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire write lock".to_string())),
        }
    }
    
    /// Loads a memory node by its ID
    pub fn load(&self, id: Uuid) -> Result<MemoryNode> {
        match self.inner.read() {
            Ok(db) => db.load(id),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire read lock".to_string())),
        }
    }
    
    /// Deletes a memory node by its ID
    pub fn delete(&self, id: Uuid) -> Result<()> {
        match self.inner.write() {
            Ok(mut db) => db.delete(id),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire write lock".to_string())),
        }
    }
    
    /// Lists all memory node IDs in the database
    pub fn list_all(&self) -> Result<Vec<Uuid>> {
        match self.inner.read() {
            Ok(db) => db.list_all(),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire read lock".to_string())),
        }
    }
    
    /// Searches for similar memories using vector similarity
    pub fn search_similar(
        &self, 
        query_vector: &[f32], 
        limit: usize, 
        threshold: f32,
        connected_to: Option<Uuid>,
        relationship_type: Option<String>,
    ) -> Result<Vec<(Uuid, f32)>> {
        match self.inner.read() {
            Ok(db) => db.search_similar(query_vector, limit, threshold, connected_to, relationship_type),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire read lock".to_string())),
        }
    }
    
    /// Returns the number of memories in the database
    pub fn len(&self) -> Result<usize> {
        match self.inner.read() {
            Ok(db) => db.len(),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire read lock".to_string())),
        }
    }
    
    /// Checks if the database is empty
    pub fn is_empty(&self) -> Result<bool> {
        match self.inner.read() {
            Ok(db) => db.is_empty(),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire read lock".to_string())),
        }
    }
    
    /// Creates a memory node from text and saves it to the database
    #[cfg(feature = "embeddings")]
    pub fn create_memory_from_text(
        &self,
        text: &str,
        embedding_service: &crate::embeddings::EmbeddingService,
        category: Option<&str>,
        attributes: Option<&std::collections::HashMap<String, crate::core::AttributeValue>>,
    ) -> Result<uuid::Uuid> {
        match self.inner.write() {
            Ok(mut db) => db.create_memory_from_text(text, embedding_service, category, attributes),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire write lock".to_string())),
        }
    }
    
    /// Creates a connection between two memory nodes
    pub fn connect(&self, source_id: Uuid, target_id: Uuid, relationship_type: String, strength: f32) -> Result<()> {
        match self.inner.write() {
            Ok(mut db) => db.connect(source_id, target_id, relationship_type, strength),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire write lock".to_string())),
        }
    }
    
    /// Removes a connection between two memory nodes
    pub fn disconnect(&self, source_id: Uuid, target_id: Uuid) -> Result<bool> {
        match self.inner.write() {
            Ok(mut db) => db.disconnect(source_id, target_id),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire write lock".to_string())),
        }
    }
    
    /// Gets all connections from a specific memory
    pub fn get_connections(&self, memory_id: Uuid, relationship_type: Option<String>) -> Result<Vec<crate::database::ConnectionInfo>> {
        match self.inner.read() {
            Ok(db) => db.get_connections(memory_id, relationship_type),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire read lock".to_string())),
        }
    }
    
    /// Gets all memories that connect to this memory
    pub fn get_connected_to(&self, memory_id: Uuid, relationship_type: Option<String>) -> Result<Vec<crate::database::ConnectionInfo>> {
        match self.inner.read() {
            Ok(db) => db.get_connected_to(memory_id, relationship_type),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire read lock".to_string())),
        }
    }
    
    /// Clears all memories and connections from the database
    pub fn clear_all(&self) -> Result<()> {
        match self.inner.write() {
            Ok(mut db) => db.clear_all(),
            Err(_) => Err(EngramDbError::Storage("Failed to acquire write lock".to_string())),
        }
    }
    
    /// Creates a query builder for this database
    ///
    /// This method creates a thread-safe query builder that can be used to
    /// construct complex queries against the database.
    ///
    /// # Returns
    ///
    /// A thread-safe query builder for this database
    pub fn query(&self) -> ThreadSafeDatabaseQueryBuilder {
        ThreadSafeDatabaseQueryBuilder {
            database: self,
        }
    }
    
    /// Gets a cloned Arc to the inner database RwLock
    ///
    /// This is an advanced method that allows direct access to the inner
    /// database instance. Use with caution.
    ///
    /// # Returns
    ///
    /// A clone of the Arc containing the RwLock-protected database
    pub fn get_inner_arc(&self) -> Arc<RwLock<Database>> {
        self.inner.clone()
    }
    
    /// Creates a new database connection pool for multi-threaded use
    ///
    /// This is useful when you need multiple database instances in different threads.
    ///
    /// # Arguments
    ///
    /// * `path` - Path to the database storage location
    ///
    /// # Returns
    ///
    /// A connection pool for creating thread-safe database connections
    pub fn create_connection_pool<P: AsRef<Path>>(path: P) -> Result<ThreadSafeDatabasePool> {
        ThreadSafeDatabasePool::new(path)
    }
}

/// A thread-safe query builder for the ThreadSafeDatabase
///
/// This struct provides a builder pattern for constructing queries against
/// a thread-safe database instance.
pub struct ThreadSafeDatabaseQueryBuilder<'a> {
    database: &'a ThreadSafeDatabase,
}

impl<'a> ThreadSafeDatabaseQueryBuilder<'a> {
    /// Sets the query vector for similarity search
    ///
    /// # Arguments
    ///
    /// * `vector` - The vector to compare against
    ///
    /// # Returns
    ///
    /// A thread-safe query with the specified vector
    pub fn with_vector(self, vector: Vec<f32>) -> ThreadSafeDatabaseQuery<'a> {
        ThreadSafeDatabaseQuery {
            builder: QueryBuilder::new().with_vector(vector),
            database: self.database,
        }
    }
    
    /// Restricts the query to only consider the specified IDs
    ///
    /// # Arguments
    ///
    /// * `ids` - The IDs to include in the query
    ///
    /// # Returns
    ///
    /// A thread-safe query restricted to the specified IDs
    pub fn with_ids(self, ids: Vec<Uuid>) -> ThreadSafeDatabaseQuery<'a> {
        ThreadSafeDatabaseQuery {
            builder: QueryBuilder::new().with_include_ids(ids),
            database: self.database,
        }
    }
    
    /// Creates an empty query
    ///
    /// # Returns
    ///
    /// An empty thread-safe query
    pub fn empty(self) -> ThreadSafeDatabaseQuery<'a> {
        ThreadSafeDatabaseQuery {
            builder: QueryBuilder::new(),
            database: self.database,
        }
    }
}

/// A thread-safe database query with builder methods for adding constraints
///
/// This struct wraps the standard QueryBuilder with thread-safe database access.
pub struct ThreadSafeDatabaseQuery<'a> {
    builder: QueryBuilder,
    database: &'a ThreadSafeDatabase,
}

impl<'a> ThreadSafeDatabaseQuery<'a> {
    /// Adds an attribute filter to the query
    ///
    /// # Arguments
    ///
    /// * `filter` - The attribute filter to add
    ///
    /// # Returns
    ///
    /// The query with the added filter
    pub fn with_attribute_filter(mut self, filter: crate::query::AttributeFilter) -> Self {
        self.builder = self.builder.with_attribute_filter(filter);
        self
    }
    
    /// Adds a temporal filter to the query
    ///
    /// # Arguments
    ///
    /// * `filter` - The temporal filter to add
    ///
    /// # Returns
    ///
    /// The query with the added filter
    pub fn with_temporal_filter(mut self, filter: crate::query::TemporalFilter) -> Self {
        self.builder = self.builder.with_temporal_filter(filter);
        self
    }
    
    /// Sets the similarity threshold for vector queries
    ///
    /// # Arguments
    ///
    /// * `threshold` - The minimum similarity threshold (0.0 to 1.0)
    ///
    /// # Returns
    ///
    /// The query with the specified threshold
    pub fn with_similarity_threshold(mut self, threshold: f32) -> Self {
        self.builder = self.builder.with_similarity_threshold(threshold);
        self
    }
    
    /// Sets the maximum number of results to return
    ///
    /// # Arguments
    ///
    /// * `limit` - The maximum number of results
    ///
    /// # Returns
    ///
    /// The query with the specified limit
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.builder = self.builder.with_limit(limit);
        self
    }
    
    /// Adds IDs to exclude from the results
    ///
    /// # Arguments
    ///
    /// * `ids` - The IDs to exclude
    ///
    /// # Returns
    ///
    /// The query with the specified exclusions
    pub fn with_exclude_ids(mut self, ids: Vec<Uuid>) -> Self {
        self.builder = self.builder.with_exclude_ids(ids);
        self
    }
    
    /// Executes the query and returns the matching memory nodes
    ///
    /// # Returns
    ///
    /// A vector of matching memory nodes, or an error if execution failed
    pub fn execute(self) -> Result<Vec<MemoryNode>> {
        // Acquire a read lock on the database
        let db_guard = match self.database.inner.read() {
            Ok(guard) => guard,
            Err(_) => return Err(EngramDbError::Storage("Failed to acquire read lock for query execution".to_string())),
        };
        
        // Execute the query using the locked database's vector index and load function
        self.builder.execute(
            db_guard.vector_index.as_ref(),
            |id| db_guard.load(id)
        )
    }
}

// Implement Clone for ThreadSafeDatabase
impl Clone for ThreadSafeDatabase {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// Implement Send and Sync for ThreadSafeDatabase
// This is safe because we use Arc<RwLock<Database>>
unsafe impl Send for ThreadSafeDatabase {}
unsafe impl Sync for ThreadSafeDatabase {}

/// A pool of thread-safe database connections
///
/// This struct manages a collection of database connections that can be
/// shared between threads. It provides methods to get a connection from the pool.
pub struct ThreadSafeDatabasePool {
    /// Path to the database
    path: Arc<Path>,
    /// Mutex to protect the connection creation
    mutex: Mutex<()>,
}

impl ThreadSafeDatabasePool {
    /// Creates a new database connection pool
    pub fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        // Test that we can create a connection
        let _test_conn = Database::file_based(&path)?;
        
        Ok(Self {
            path: Arc::from(path.as_ref()),
            mutex: Mutex::new(()),
        })
    }
    
    /// Get a connection from the pool
    ///
    /// This will create a new connection if needed.
    pub fn get_connection(&self) -> Result<ThreadSafeDatabase> {
        // Acquire the mutex to protect connection creation
        let _guard = self.mutex.lock().map_err(|_| {
            EngramDbError::Storage("Failed to acquire mutex for connection pool".to_string())
        })?;
        
        // Create a new connection
        ThreadSafeDatabase::file_based(self.path.as_ref())
    }
}

// Implement Clone for ThreadSafeDatabasePool
impl Clone for ThreadSafeDatabasePool {
    fn clone(&self) -> Self {
        Self {
            path: self.path.clone(),
            mutex: Mutex::new(()),
        }
    }
}

// Implement Send and Sync for ThreadSafeDatabasePool
// This is safe because we use Arc and Mutex
unsafe impl Send for ThreadSafeDatabasePool {}
unsafe impl Sync for ThreadSafeDatabasePool {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Barrier};
    use std::thread;
    
    #[test]
    fn test_thread_safe_database_basic() {
        let db = ThreadSafeDatabase::in_memory();
        
        // Create and save a memory node
        let node = MemoryNode::new(vec![1.0, 0.0, 0.0]);
        let id = db.save(&node).unwrap();
        
        // Load the node back
        let loaded = db.load(id).unwrap();
        assert_eq!(loaded.id(), node.id());
        assert_eq!(loaded.embeddings(), node.embeddings());
        
        // Test list_all
        let all_ids = db.list_all().unwrap();
        assert_eq!(all_ids.len(), 1);
        assert_eq!(all_ids[0], id);
        
        // Test delete
        db.delete(id).unwrap();
        assert!(db.is_empty().unwrap());
    }
    
    #[test]
    fn test_thread_safe_database_concurrent() {
        let db = Arc::new(ThreadSafeDatabase::in_memory());
        let num_threads = 8;
        let num_ops = 100;
        
        // Create a barrier to synchronize threads
        let barrier = Arc::new(Barrier::new(num_threads));
        
        // Create threads
        let mut handles = Vec::with_capacity(num_threads);
        
        for thread_id in 0..num_threads {
            let db_clone = db.clone();
            let barrier_clone = barrier.clone();
            
            let handle = thread::spawn(move || {
                // Wait for all threads to be ready
                barrier_clone.wait();
                
                // Perform operations
                for i in 0..num_ops {
                    let node_id = (thread_id * num_ops) + i;
                    let node = MemoryNode::new(vec![node_id as f32, 0.0, 0.0]);
                    let saved_id = db_clone.save(&node).unwrap();
                    
                    // Verify save worked
                    let loaded = db_clone.load(saved_id).unwrap();
                    assert_eq!(loaded.id(), node.id());
                }
                
                // Return the number of operations performed
                num_ops
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        let mut total_ops = 0;
        for handle in handles {
            total_ops += handle.join().unwrap();
        }
        
        // Verify total operations
        assert_eq!(total_ops, num_threads * num_ops);
        
        // Verify database state
        let all_ids = db.list_all().unwrap();
        assert_eq!(all_ids.len(), total_ops);
    }
    
    #[test]
    fn test_connection_pool() {
        use tempfile::tempdir;
        
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_pool.db");
        
        // Create the connection pool
        let pool = ThreadSafeDatabasePool::new(&path).unwrap();
        
        // Get connections from the pool
        let conn1 = pool.get_connection().unwrap();
        let conn2 = pool.get_connection().unwrap();
        
        // Test independent operations
        let node1 = MemoryNode::new(vec![1.0, 0.0, 0.0]);
        let node2 = MemoryNode::new(vec![0.0, 1.0, 0.0]);
        
        let id1 = conn1.save(&node1).unwrap();
        let id2 = conn2.save(&node2).unwrap();
        
        // Verify both connections can access the same data
        assert_eq!(conn1.list_all().unwrap().len(), 2);
        assert_eq!(conn2.list_all().unwrap().len(), 2);
        
        assert!(conn1.load(id2).is_ok());
        assert!(conn2.load(id1).is_ok());
    }
    
    #[test]
    fn test_multi_threaded_connection_pool() {
        use tempfile::tempdir;
        
        // Create a temporary directory
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_mt_pool.db");
        
        // Create the connection pool
        let pool = Arc::new(ThreadSafeDatabasePool::new(&path).unwrap());
        let num_threads = 8;
        let num_ops = 50;
        
        // Create a barrier to synchronize threads
        let barrier = Arc::new(Barrier::new(num_threads));
        
        // Create threads
        let mut handles = Vec::with_capacity(num_threads);
        
        for thread_id in 0..num_threads {
            let pool_clone = pool.clone();
            let barrier_clone = barrier.clone();
            
            let handle = thread::spawn(move || {
                // Get a connection from the pool
                let conn = pool_clone.get_connection().unwrap();
                
                // Wait for all threads to be ready
                barrier_clone.wait();
                
                // Perform operations
                for i in 0..num_ops {
                    let node_id = (thread_id * num_ops) + i;
                    let node = MemoryNode::new(vec![node_id as f32, 0.0, 0.0]);
                    let saved_id = conn.save(&node).unwrap();
                    
                    // Verify save worked
                    let loaded = conn.load(saved_id).unwrap();
                    assert_eq!(loaded.id(), node.id());
                }
                
                // Return the number of operations performed
                num_ops
            });
            
            handles.push(handle);
        }
        
        // Wait for all threads to complete
        let mut total_ops = 0;
        for handle in handles {
            total_ops += handle.join().unwrap();
        }
        
        // Verify total operations
        assert_eq!(total_ops, num_threads * num_ops);
        
        // Get a final connection to verify state
        let final_conn = pool.get_connection().unwrap();
        let all_ids = final_conn.list_all().unwrap();
        assert_eq!(all_ids.len(), total_ops);
    }
}