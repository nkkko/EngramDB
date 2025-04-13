//! Thread-safe implementations for EngramDB
//!
//! This module provides thread-safe wrappers for the Database and related components.

use std::sync::{Arc, RwLock, Mutex};
use crate::core::MemoryNode;
use crate::database::Database;
use crate::error::EngramDbError;
use crate::Result;
use std::path::Path;
use uuid::Uuid;

/// A thread-safe wrapper for the EngramDB Database
///
/// This struct wraps the Database with synchronization primitives to make it
/// safe to share between threads. It implements Send and Sync traits.
pub struct ThreadSafeDatabase {
    /// The inner database protected by a RwLock for concurrent access
    inner: Arc<RwLock<Database>>,
}

impl ThreadSafeDatabase {
    /// Creates a new thread-safe in-memory database
    pub fn in_memory() -> Self {
        let db = Database::in_memory();
        Self {
            inner: Arc::new(RwLock::new(db)),
        }
    }
    
    /// Creates a new thread-safe in-memory database with HNSW index
    pub fn in_memory_with_hnsw() -> Self {
        let db = Database::in_memory_with_hnsw();
        Self {
            inner: Arc::new(RwLock::new(db)),
        }
    }
    
    /// Creates a new thread-safe file-based database
    pub fn file_based<P: AsRef<Path>>(dir: P) -> Result<Self> {
        match Database::file_based(dir) {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
        }
    }
    
    /// Creates a new thread-safe single-file database
    pub fn single_file<P: AsRef<Path>>(file_path: P) -> Result<Self> {
        match Database::single_file(file_path) {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
        }
    }
    
    /// Creates a thread-safe file-based database with HNSW index
    pub fn file_based_with_hnsw<P: AsRef<Path>>(path: P) -> Result<Self> {
        match Database::file_based_with_hnsw(path) {
            Ok(db) => Ok(Self {
                inner: Arc::new(RwLock::new(db)),
            }),
            Err(e) => Err(e),
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
    
    /// Creates a new database connection pool for multi-threaded use
    ///
    /// This is useful when you need multiple database instances in different threads.
    pub fn create_connection_pool<P: AsRef<Path>>(path: P) -> Result<ThreadSafeDatabasePool> {
        Ok(ThreadSafeDatabasePool::new(path)?)
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