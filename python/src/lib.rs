use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1};
use uuid::Uuid;
use engramdb::{
    MemoryNode as EngramDbMemoryNode,
    Database as EngramDbDatabase,
    RelationshipType as EngramDbRelationshipType,
    ThreadSafeDatabase as EngramDbThreadSafeDatabase,
    ThreadSafeDatabasePool as EngramDbThreadSafeDatabasePool,
    DatabaseConfig as EngramDbDatabaseConfig,
    StorageType as EngramDbStorageType,
    EmbeddingModel as EngramDbEmbeddingModel,
};

// Vector and multi-vector related imports
use engramdb::vector::{
    MultiVectorIndexConfig as EngramDbMultiVectorIndexConfig,
    MultiVectorSimilarityMethod as EngramDbMultiVectorSimilarityMethod,
    VectorAlgorithm as EngramDbVectorAlgorithm,
};

#[cfg(feature = "embeddings")]
use engramdb::{
    EmbeddingService as EngramDbEmbeddingService,
    embeddings::multi_vector::{
        MultiVectorEmbedding as EngramDbMultiVectorEmbedding,
        MultiVectorProvider as EngramDbMultiVectorProvider,
    },
};
use engramdb::core::AttributeValue;
use std::path::PathBuf;

// Import the sample dataset module
#[path = "../../examples/rust/sample_dataset.rs"]
mod sample_dataset;

/// Simple function to test that the Python binding works
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> String {
    format!("{}", a + b)
}

/// A Python wrapper for the EngramDB MemoryNode class
#[pyclass]
struct MemoryNode {
    inner: EngramDbMemoryNode,
}

#[pymethods]
impl MemoryNode {
    /// Create a new memory node with the given embeddings
    #[new]
    fn new(embeddings: Vec<f32>) -> Self {
        Self {
            inner: EngramDbMemoryNode::new(embeddings),
        }
    }
    
    #[cfg(feature = "embeddings")]
    #[staticmethod]
    fn from_text(text: &str, embedding_service: &EmbeddingService, category: Option<&str>) -> PyResult<Self> {
        match EngramDbMemoryNode::from_text(text, &embedding_service.inner, category) {
            Ok(node) => Ok(Self { inner: node }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create memory from text: {}", e))),
        }
    }

    /// Get the ID of the memory node
    #[getter]
    fn id(&self) -> String {
        self.inner.id().to_string()
    }

    /// Set an attribute on the memory node
    fn set_attribute(&mut self, key: &str, value: &PyAny) -> PyResult<()> {
        let attr_value = match value.extract::<String>() {
            Ok(s) => AttributeValue::String(s),
            Err(_) => match value.extract::<i64>() {
                Ok(i) => AttributeValue::Integer(i),
                Err(_) => match value.extract::<f64>() {
                    Ok(f) => AttributeValue::Float(f),
                    Err(_) => match value.extract::<bool>() {
                        Ok(b) => AttributeValue::Boolean(b),
                        Err(_) => return Err(PyValueError::new_err("Unsupported attribute value type")),
                    },
                },
            },
        };
        
        self.inner.set_attribute(key.to_string(), attr_value);
        Ok(())
    }

    /// Get an attribute from the memory node
    fn get_attribute(&self, key: &str) -> Option<PyObject> {
        Python::with_gil(|py| {
            self.inner.get_attribute(key).map(|attr| {
                match attr {
                    AttributeValue::String(s) => s.to_object(py),
                    AttributeValue::Integer(i) => i.to_object(py),
                    AttributeValue::Float(f) => f.to_object(py),
                    AttributeValue::Boolean(b) => b.to_object(py),
                    _ => py.None(),
                }
            })
        })
    }
    
    /// Get all attributes of the memory node
    fn attributes<'py>(&self, py: Python<'py>) -> PyObject {
        let dict = pyo3::types::PyDict::new(py);
        for (key, value) in self.inner.attributes() {
            let py_value = match value {
                AttributeValue::String(s) => s.to_object(py),
                AttributeValue::Integer(i) => i.to_object(py),
                AttributeValue::Float(f) => f.to_object(py),
                AttributeValue::Boolean(b) => b.to_object(py),
                _ => py.None(),
            };
            dict.set_item(key, py_value).unwrap();
        }
        dict.to_object(py)
    }

    /// Get the embeddings of this memory node
    fn get_embeddings<'py>(&self, py: Python<'py>) -> Option<&'py PyArray1<f32>> {
        self.inner.embeddings().map(|v| v.to_vec().into_pyarray(py))
    }
    
    /// Get the legacy embeddings (flattened if multi-vector)
    fn get_embeddings_legacy<'py>(&self, py: Python<'py>) -> &'py PyArray1<f32> {
        self.inner.embeddings_legacy().into_pyarray(py)
    }
    
    /// Set new embeddings for this memory node
    fn set_embeddings(&mut self, embeddings: Vec<f32>) {
        self.inner.set_embeddings(embeddings);
    }
    
    /// Create a memory node with multi-vector embeddings
    #[cfg(feature = "embeddings")]
    #[staticmethod]
    fn with_multi_vector(multi_vector: &MultiVectorEmbedding) -> Self {
        Self {
            inner: EngramDbMemoryNode::with_multi_vector(multi_vector.inner.clone()),
        }
    }
    
    /// Create a memory node from text using multi-vector embeddings
    #[cfg(feature = "embeddings")]
    #[staticmethod]
    fn from_text_multi_vector(text: &str, embedding_service: &EmbeddingService, category: Option<&str>) -> PyResult<Self> {
        if !embedding_service.inner.has_multi_vector() {
            return Err(PyValueError::new_err("Embedding service does not have multi-vector capability"));
        }
        
        match embedding_service.inner.generate_multi_vector_for_document(text, category) {
            Ok(multi_vec) => {
                let node = EngramDbMemoryNode::with_multi_vector(multi_vec);
                // Add the text content as an attribute
                let mut new_node = node.clone();
                new_node.set_attribute("content".to_string(), AttributeValue::String(text.to_string()));
                
                // Add category if provided
                if let Some(cat) = category {
                    new_node.set_attribute("category".to_string(), AttributeValue::String(cat.to_string()));
                }
                
                Ok(Self { inner: new_node })
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to create multi-vector memory: {}", e)))
        }
    }
    
    /// Check if this memory node uses multi-vector embeddings
    fn is_multi_vector(&self) -> bool {
        self.inner.is_multi_vector()
    }
    
    /// Get multi-vector embeddings if available
    #[cfg(feature = "embeddings")]
    fn multi_vector_embeddings(&self) -> Option<MultiVectorEmbedding> {
        self.inner.multi_vector_embeddings().map(|mv| MultiVectorEmbedding { inner: mv.clone() })
    }
}

/// Python wrapper for the EngramDB Database
#[pyclass(unsendable)]
struct Database {
    inner: EngramDbDatabase,
}

#[cfg(feature = "embeddings")]
#[pyclass]
struct EmbeddingService {
    inner: EngramDbEmbeddingService,
}

#[pymethods]
impl Database {
    /// Create a new in-memory database
    #[staticmethod]
    fn in_memory() -> Self {
        Self {
            inner: EngramDbDatabase::in_memory(),
        }
    }
    
    /// Create a new in-memory database with HNSW index
    #[staticmethod]
    fn in_memory_with_hnsw() -> Self {
        Self {
            inner: EngramDbDatabase::in_memory_with_hnsw(),
        }
    }
    
    /// Create a new in-memory database with multi-vector index
    #[staticmethod]
    fn in_memory_with_multi_vector() -> Self {
        Self {
            inner: EngramDbDatabase::in_memory_with_multi_vector(),
        }
    }
    
    /// Create a database with a specific configuration
    #[staticmethod]
    fn with_config(config: &DatabaseConfig) -> PyResult<Self> {
        match EngramDbDatabase::with_config(config.inner.clone()) {
            Ok(db) => Ok(Self { inner: db }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create database with config: {}", e))),
        }
    }

    /// Create a new file-based database at the given path
    #[staticmethod]
    fn file_based(path: &str) -> PyResult<Self> {
        match EngramDbDatabase::file_based(PathBuf::from(path)) {
            Ok(db) => Ok(Self { inner: db }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create file-based database: {}", e)))
        }
    }
    
    /// Create a file-based database with the specified path and HNSW index
    #[staticmethod]
    fn file_based_with_hnsw(path: &str) -> PyResult<Self> {
        match EngramDbDatabase::file_based_with_hnsw(PathBuf::from(path)) {
            Ok(db) => Ok(Self { inner: db }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create database with HNSW: {}", e)))
        }
    }
    
    /// Create a file-based database with the specified path and multi-vector index
    #[staticmethod]
    fn file_based_with_multi_vector(path: &str) -> PyResult<Self> {
        match EngramDbDatabase::file_based_with_multi_vector(PathBuf::from(path)) {
            Ok(db) => Ok(Self { inner: db }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create database with multi-vector: {}", e)))
        }
    }

    /// Save a memory node to the database
    fn save(&mut self, memory_node: &MemoryNode) -> PyResult<String> {
        match self.inner.save(&memory_node.inner) {
            Ok(id) => Ok(id.to_string()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to save node: {}", e)))
        }
    }

    /// Load a memory node from the database
    fn load(&self, id_str: &str) -> PyResult<MemoryNode> {
        let id = Uuid::parse_str(id_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.load(id) {
            Ok(node) => Ok(MemoryNode { inner: node }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to load node: {}", e)))
        }
    }

    /// Delete a memory node from the database
    fn delete(&mut self, id_str: &str) -> PyResult<bool> {
        let id = Uuid::parse_str(id_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.delete(id) {
            Ok(_) => Ok(true),
            Err(e) => Err(PyValueError::new_err(format!("Failed to delete node: {}", e)))
        }
    }

    /// List all memory node IDs in the database
    fn list_all(&self) -> PyResult<Vec<String>> {
        match self.inner.list_all() {
            Ok(ids) => Ok(ids.into_iter().map(|id| id.to_string()).collect()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to list nodes: {}", e)))
        }
    }
    
    #[cfg(feature = "embeddings")]
    fn create_memory_from_text(
        &mut self, 
        text: &str, 
        embedding_service: &EmbeddingService, 
        category: Option<&str>, 
        title: Option<&str>,
    ) -> PyResult<String> {
        // Create attribute dictionary if title is provided
        let mut attributes = std::collections::HashMap::new();
        if let Some(t) = title {
            attributes.insert("title".to_string(), AttributeValue::String(t.to_string()));
        }
        
        match self.inner.create_memory_from_text(
            text,
            &embedding_service.inner,
            category,
            if attributes.is_empty() { None } else { Some(&attributes) },
        ) {
            Ok(id) => Ok(id.to_string()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create memory from text: {}", e))),
        }
    }

    /// Search for similar memory nodes using vector similarity
    #[pyo3(signature = (query, limit, threshold, connected_to=None, relationship_type=None))]
    fn search_similar(&self, query: Vec<f32>, limit: usize, threshold: f32, 
                      connected_to: Option<&str>, relationship_type: Option<&str>) -> PyResult<Vec<(String, f32)>> {
        // Parse the connected_to UUID if provided
        let connected_id = if let Some(id_str) = connected_to {
            Some(Uuid::parse_str(id_str)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?)
        } else {
            None
        };
        
        match self.inner.search_similar(&query, limit, threshold, connected_id, relationship_type.map(|s| s.to_string())) {
            Ok(results) => Ok(results.into_iter().map(|(id, score)| (id.to_string(), score)).collect()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to search: {}", e)))
        }
    }
    
    /// Create a connection between two memory nodes
    fn connect(&mut self, source_id: &str, target_id: &str, relationship_type: &str, strength: f32) -> PyResult<()> {
        let source = Uuid::parse_str(source_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid source UUID: {}", e)))?;
            
        let target = Uuid::parse_str(target_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid target UUID: {}", e)))?;
        
        match self.inner.connect(source, target, relationship_type.to_string(), strength) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create connection: {}", e)))
        }
    }
    
    /// Remove a connection between two memory nodes
    fn disconnect(&mut self, source_id: &str, target_id: &str) -> PyResult<bool> {
        let source = Uuid::parse_str(source_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid source UUID: {}", e)))?;
            
        let target = Uuid::parse_str(target_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid target UUID: {}", e)))?;
        
        match self.inner.disconnect(source, target) {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!("Failed to remove connection: {}", e)))
        }
    }
    
    /// Get all connections from a memory node
    #[pyo3(signature = (memory_id, relationship_type=None))]
    fn get_connections(&self, memory_id: &str, relationship_type: Option<&str>) -> PyResult<Vec<PyObject>> {
        let id = Uuid::parse_str(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.get_connections(id, relationship_type.map(|s| s.to_string())) {
            Ok(connections) => {
                Python::with_gil(|py| {
                    Ok(connections.into_iter().map(|conn| {
                        let dict = pyo3::types::PyDict::new(py);
                        dict.set_item("target_id", conn.target_id.to_string()).unwrap();
                        dict.set_item("type", conn.type_name).unwrap();
                        dict.set_item("strength", conn.strength).unwrap();
                        dict.to_object(py)
                    }).collect())
                })
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to get connections: {}", e)))
        }
    }
    
    /// Get all memories that connect to this memory
    #[pyo3(signature = (memory_id, relationship_type=None))]
    fn get_connected_to(&self, memory_id: &str, relationship_type: Option<&str>) -> PyResult<Vec<PyObject>> {
        let id = Uuid::parse_str(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.get_connected_to(id, relationship_type.map(|s| s.to_string())) {
            Ok(connections) => {
                Python::with_gil(|py| {
                    Ok(connections.into_iter().map(|conn| {
                        let dict = pyo3::types::PyDict::new(py);
                        dict.set_item("source_id", conn.target_id.to_string()).unwrap();
                        dict.set_item("type", conn.type_name).unwrap();
                        dict.set_item("strength", conn.strength).unwrap();
                        dict.to_object(py)
                    }).collect())
                })
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to get connected_to: {}", e)))
        }
    }
    
    /// Clear all memories and connections
    fn clear_all(&mut self) -> PyResult<()> {
        match self.inner.clear_all() {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to clear database: {}", e)))
        }
    }
    
    /// Create a memory node from text using multi-vector embeddings
    #[cfg(feature = "embeddings")]
    fn create_memory_from_text_multi_vector(
        &mut self,
        text: &str,
        embedding_service: &EmbeddingService,
        category: Option<&str>,
        title: Option<&str>,
    ) -> PyResult<String> {
        // Create attribute dictionary if title is provided
        let mut attributes = std::collections::HashMap::new();
        if let Some(t) = title {
            attributes.insert("title".to_string(), AttributeValue::String(t.to_string()));
        }
        
        // Use the EngramDB database to create the memory with multi-vector
        match self.inner.create_memory_from_text_multi_vector(
            text,
            &embedding_service.inner,
            category,
            if attributes.is_empty() { None } else { Some(&attributes) }
        ) {
            Ok(id) => Ok(id.to_string()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create multi-vector memory: {}", e))),
        }
    }
    
    /// Perform a search using a vector query and return memory nodes
    fn search_by_vector(&self, query_vector: Vec<f32>, limit: usize, threshold: f32) -> PyResult<Vec<(MemoryNode, f32)>> {
        match self.inner.search_by_vector(&query_vector, limit, threshold) {
            Ok(results) => {
                let mut memory_results = Vec::with_capacity(results.len());
                for (node, score) in results {
                    memory_results.push((MemoryNode { inner: node }, score));
                }
                Ok(memory_results)
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to search by vector: {}", e))),
        }
    }
    
    /// Perform a search using a multi-vector query and return memory nodes
    #[cfg(feature = "embeddings")]
    fn search_by_multi_vector(&self, multi_vector: &MultiVectorEmbedding, limit: usize, threshold: f32) -> PyResult<Vec<(MemoryNode, f32)>> {
        match self.inner.search_by_multi_vector(&multi_vector.inner, limit, threshold) {
            Ok(results) => {
                let mut memory_results = Vec::with_capacity(results.len());
                for (node, score) in results {
                    memory_results.push((MemoryNode { inner: node }, score));
                }
                Ok(memory_results)
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to search by multi-vector: {}", e))),
        }
    }
}

/// Python enum for relationship types between memory nodes
#[pyclass]
#[derive(Clone, Copy)]
enum RelationshipType {
    ASSOCIATION = 0,
    CAUSATION = 1,
    SEQUENCE = 2,
    CONTAINS = 3,
    PartOf = 4,
    PREDECESSOR = 5,
    SUCCESSOR = 6,
    REFERENCE = 7,
}

impl From<RelationshipType> for EngramDbRelationshipType {
    fn from(py_type: RelationshipType) -> Self {
        match py_type {
            RelationshipType::ASSOCIATION => EngramDbRelationshipType::Association,
            RelationshipType::CAUSATION => EngramDbRelationshipType::Causation,
            RelationshipType::SEQUENCE => EngramDbRelationshipType::Sequence,
            RelationshipType::CONTAINS => EngramDbRelationshipType::Contains,
            RelationshipType::PartOf => EngramDbRelationshipType::PartOf,
            RelationshipType::PREDECESSOR => EngramDbRelationshipType::Custom("Predecessor".to_string()),
            RelationshipType::SUCCESSOR => EngramDbRelationshipType::Custom("Successor".to_string()),
            RelationshipType::REFERENCE => EngramDbRelationshipType::Custom("Reference".to_string()),
        }
    }
}

#[cfg(feature = "embeddings")]
#[pyclass]
enum EmbeddingModelType {
    E5 = 0,
    GTE = 1,
    JINA = 2,
    JINA_COLBERT = 3,
    CUSTOM = 4,
}

#[cfg(feature = "embeddings")]
impl From<EmbeddingModelType> for engramdb::embeddings::EmbeddingModel {
    fn from(model_type: EmbeddingModelType) -> Self {
        match model_type {
            EmbeddingModelType::E5 => engramdb::embeddings::EmbeddingModel::E5MultilingualLargeInstruct,
            EmbeddingModelType::GTE => engramdb::embeddings::EmbeddingModel::GteModernBertBase,
            EmbeddingModelType::JINA => engramdb::embeddings::EmbeddingModel::JinaEmbeddingsV3,
            EmbeddingModelType::JINA_COLBERT => engramdb::embeddings::EmbeddingModel::JinaColBERTv2,
            EmbeddingModelType::CUSTOM => engramdb::embeddings::EmbeddingModel::Custom,
        }
    }
}

#[cfg(feature = "embeddings")]
#[pymethods]
impl EmbeddingService {
    /// Create a new embedding service with default configuration
    #[staticmethod]
    fn default() -> Self {
        Self {
            inner: EngramDbEmbeddingService::default(),
        }
    }
    
    /// Create a new embedding service using a mock provider
    #[staticmethod]
    fn mock(dimensions: usize) -> Self {
        Self {
            inner: EngramDbEmbeddingService::new_mock(dimensions),
        }
    }
    
    /// Create a new embedding service using a model provider with a custom model name
    #[staticmethod]
    fn with_model(model_name: Option<&str>) -> Self {
        Self {
            inner: EngramDbEmbeddingService::new_with_model(model_name),
        }
    }
    
    /// Create a new embedding service using a specific pre-defined model type
    #[staticmethod]
    fn with_model_type(model_type: EmbeddingModelType) -> Self {
        let rust_model_type = engramdb::embeddings::EmbeddingModel::from(model_type);
        Self {
            inner: EngramDbEmbeddingService::with_model_type(rust_model_type),
        }
    }
    
    /// Get the dimensions of the embeddings
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
    
    /// Generate embeddings for a document
    fn generate_for_document(&self, text: &str, category: Option<&str>) -> PyResult<Vec<f32>> {
        match self.inner.generate_for_document(text, category) {
            Ok(embeddings) => Ok(embeddings),
            Err(e) => Err(PyValueError::new_err(format!("Failed to generate embeddings: {}", e))),
        }
    }
    
    /// Generate embeddings for a query
    fn generate_for_query(&self, text: &str) -> PyResult<Vec<f32>> {
        match self.inner.generate_for_query(text) {
            Ok(embeddings) => Ok(embeddings),
            Err(e) => Err(PyValueError::new_err(format!("Failed to generate embeddings: {}", e))),
        }
    }
    
    /// Generate random embeddings
    fn generate_random(&self) -> Vec<f32> {
        self.inner.generate_random()
    }
    
    /// Returns whether this service has multi-vector capability
    fn has_multi_vector(&self) -> bool {
        self.inner.has_multi_vector()
    }
    
    /// Returns the dimensions of multi-vector embeddings, if available
    fn multi_vector_dimensions(&self) -> Option<usize> {
        self.inner.multi_vector_dimensions()
    }
    
    /// Create a new embedding service with a multi-vector model
    #[staticmethod]
    fn with_multi_vector_model(model_type: EmbeddingModelType) -> Self {
        let rust_model_type = EngramDbEmbeddingModel::from(model_type);
        Self {
            inner: EngramDbEmbeddingService::with_multi_vector_model(rust_model_type),
        }
    }
    
    /// Create a new embedding service with mock providers for both single and multi-vector
    #[staticmethod]
    fn new_mock_multi_vector(dimensions: usize, num_vectors: usize) -> Self {
        Self {
            inner: EngramDbEmbeddingService::new_mock_multi_vector(dimensions, num_vectors),
        }
    }
    
    /// Generate multi-vector embeddings for text
    fn generate_multi_vector(&self, text: &str) -> PyResult<MultiVectorEmbedding> {
        match self.inner.generate_multi_vector(text) {
            Ok(embeddings) => Ok(MultiVectorEmbedding { inner: embeddings }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to generate multi-vector embeddings: {}", e))),
        }
    }
    
    /// Generate multi-vector embeddings for a document
    fn generate_multi_vector_for_document(&self, text: &str, category: Option<&str>) -> PyResult<MultiVectorEmbedding> {
        match self.inner.generate_multi_vector_for_document(text, category) {
            Ok(embeddings) => Ok(MultiVectorEmbedding { inner: embeddings }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to generate multi-vector embeddings: {}", e))),
        }
    }
    
    /// Generate multi-vector embeddings for a query
    fn generate_multi_vector_for_query(&self, text: &str) -> PyResult<MultiVectorEmbedding> {
        match self.inner.generate_multi_vector_for_query(text) {
            Ok(embeddings) => Ok(MultiVectorEmbedding { inner: embeddings }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to generate multi-vector embeddings: {}", e))),
        }
    }
    
    /// Generate random multi-vector embeddings
    fn generate_random_multi_vector(&self, num_vectors: Option<usize>) -> Option<MultiVectorEmbedding> {
        self.inner.generate_random_multi_vector(num_vectors)
            .map(|embeddings| MultiVectorEmbedding { inner: embeddings })
    }
}

/// Load a sample dataset with an AI Coding Agent bug fixing workflow
#[pyfunction]
fn load_sample_dataset(db: &mut Database) -> PyResult<Vec<String>> {
    match sample_dataset::load_sample_dataset(&mut db.inner) {
        Ok(ids) => Ok(ids.into_iter().map(|id| id.to_string()).collect()),
        Err(e) => Err(PyValueError::new_err(format!("Failed to load sample dataset: {}", e)))
    }
}

/// Load a minimal sample dataset for demonstration purposes
#[pyfunction]
fn load_minimal_dataset(db: &mut Database) -> PyResult<Vec<String>> {
    match sample_dataset::load_minimal_dataset(&mut db.inner) {
        Ok(ids) => Ok(ids.into_iter().map(|id| id.to_string()).collect()),
        Err(e) => Err(PyValueError::new_err(format!("Failed to load minimal dataset: {}", e)))
    }
}

/// Python wrapper for thread-safe database
#[pyclass]
struct ThreadSafeDatabase {
    inner: EngramDbThreadSafeDatabase,
}

/// Python wrapper for thread-safe database connection pool
#[pyclass]
struct ThreadSafeDatabasePool {
    inner: EngramDbThreadSafeDatabasePool,
}

#[pymethods]
impl ThreadSafeDatabase {
    /// Create a new in-memory thread-safe database
    #[staticmethod]
    fn in_memory() -> Self {
        Self {
            inner: EngramDbThreadSafeDatabase::in_memory(),
        }
    }

    /// Create a new file-based thread-safe database at the given path
    #[staticmethod]
    fn file_based(path: &str) -> PyResult<Self> {
        match EngramDbThreadSafeDatabase::file_based(PathBuf::from(path)) {
            Ok(db) => Ok(Self { inner: db }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create thread-safe file-based database: {}", e)))
        }
    }
    
    /// Create a new in-memory thread-safe database with HNSW index
    #[staticmethod]
    fn in_memory_with_hnsw() -> Self {
        Self {
            inner: EngramDbThreadSafeDatabase::in_memory_with_hnsw(),
        }
    }

    /// Save a memory node to the database
    fn save(&self, memory_node: &MemoryNode) -> PyResult<String> {
        match self.inner.save(&memory_node.inner) {
            Ok(id) => Ok(id.to_string()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to save node: {}", e)))
        }
    }

    /// Load a memory node from the database
    fn load(&self, id_str: &str) -> PyResult<MemoryNode> {
        let id = Uuid::parse_str(id_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.load(id) {
            Ok(node) => Ok(MemoryNode { inner: node }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to load node: {}", e)))
        }
    }

    /// Delete a memory node from the database
    fn delete(&self, id_str: &str) -> PyResult<bool> {
        let id = Uuid::parse_str(id_str)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.delete(id) {
            Ok(_) => Ok(true),
            Err(e) => Err(PyValueError::new_err(format!("Failed to delete node: {}", e)))
        }
    }

    /// List all memory node IDs in the database
    fn list_all(&self) -> PyResult<Vec<String>> {
        match self.inner.list_all() {
            Ok(ids) => Ok(ids.into_iter().map(|id| id.to_string()).collect()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to list nodes: {}", e)))
        }
    }
    
    /// Search for similar memory nodes using vector similarity
    #[pyo3(signature = (query, limit, threshold, connected_to=None, relationship_type=None))]
    fn search_similar(&self, query: Vec<f32>, limit: usize, threshold: f32, 
                      connected_to: Option<&str>, relationship_type: Option<&str>) -> PyResult<Vec<(String, f32)>> {
        // Parse the connected_to UUID if provided
        let connected_id = if let Some(id_str) = connected_to {
            Some(Uuid::parse_str(id_str)
                .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?)
        } else {
            None
        };
        
        match self.inner.search_similar(&query, limit, threshold, connected_id, relationship_type.map(|s| s.to_string())) {
            Ok(results) => Ok(results.into_iter().map(|(id, score)| (id.to_string(), score)).collect()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to search: {}", e)))
        }
    }
    
    /// Create a connection between two memory nodes
    fn connect(&self, source_id: &str, target_id: &str, relationship_type: &str, strength: f32) -> PyResult<()> {
        let source = Uuid::parse_str(source_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid source UUID: {}", e)))?;
            
        let target = Uuid::parse_str(target_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid target UUID: {}", e)))?;
        
        match self.inner.connect(source, target, relationship_type.to_string(), strength) {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create connection: {}", e)))
        }
    }
    
    /// Remove a connection between two memory nodes
    fn disconnect(&self, source_id: &str, target_id: &str) -> PyResult<bool> {
        let source = Uuid::parse_str(source_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid source UUID: {}", e)))?;
            
        let target = Uuid::parse_str(target_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid target UUID: {}", e)))?;
        
        match self.inner.disconnect(source, target) {
            Ok(result) => Ok(result),
            Err(e) => Err(PyValueError::new_err(format!("Failed to remove connection: {}", e)))
        }
    }
    
    /// Get all connections from a memory node
    #[pyo3(signature = (memory_id, relationship_type=None))]
    fn get_connections(&self, memory_id: &str, relationship_type: Option<&str>) -> PyResult<Vec<PyObject>> {
        let id = Uuid::parse_str(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.get_connections(id, relationship_type.map(|s| s.to_string())) {
            Ok(connections) => {
                Python::with_gil(|py| {
                    Ok(connections.into_iter().map(|conn| {
                        let dict = pyo3::types::PyDict::new(py);
                        dict.set_item("target_id", conn.target_id.to_string()).unwrap();
                        dict.set_item("type", conn.type_name).unwrap();
                        dict.set_item("strength", conn.strength).unwrap();
                        dict.to_object(py)
                    }).collect())
                })
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to get connections: {}", e)))
        }
    }
    
    /// Get all memories that connect to this memory
    #[pyo3(signature = (memory_id, relationship_type=None))]
    fn get_connected_to(&self, memory_id: &str, relationship_type: Option<&str>) -> PyResult<Vec<PyObject>> {
        let id = Uuid::parse_str(memory_id)
            .map_err(|e| PyValueError::new_err(format!("Invalid UUID: {}", e)))?;
        
        match self.inner.get_connected_to(id, relationship_type.map(|s| s.to_string())) {
            Ok(connections) => {
                Python::with_gil(|py| {
                    Ok(connections.into_iter().map(|conn| {
                        let dict = pyo3::types::PyDict::new(py);
                        dict.set_item("source_id", conn.target_id.to_string()).unwrap();
                        dict.set_item("type", conn.type_name).unwrap();
                        dict.set_item("strength", conn.strength).unwrap();
                        dict.to_object(py)
                    }).collect())
                })
            },
            Err(e) => Err(PyValueError::new_err(format!("Failed to get connected_to: {}", e)))
        }
    }
    
    /// Clear all memories and connections
    fn clear_all(&self) -> PyResult<()> {
        match self.inner.clear_all() {
            Ok(_) => Ok(()),
            Err(e) => Err(PyValueError::new_err(format!("Failed to clear database: {}", e)))
        }
    }
}

/// Implementation of ThreadSafeDatabasePool Python bindings
#[pymethods]
impl ThreadSafeDatabasePool {
    /// Create a new database connection pool
    #[staticmethod]
    fn new(path: &str) -> PyResult<Self> {
        match EngramDbThreadSafeDatabasePool::new(PathBuf::from(path)) {
            Ok(pool) => Ok(Self { inner: pool }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create database pool: {}", e)))
        }
    }
    
    /// Get a connection from the pool
    fn get_connection(&self) -> PyResult<ThreadSafeDatabase> {
        match self.inner.get_connection() {
            Ok(db) => Ok(ThreadSafeDatabase { inner: db }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to get connection from pool: {}", e)))
        }
    }
}

/// Python enum for vector algorithm types
#[pyclass]
#[derive(Clone, Copy)]
enum VectorAlgorithm {
    Linear = 0,
    HNSW = 1,
    MultiVector = 2,
}

impl From<VectorAlgorithm> for EngramDbVectorAlgorithm {
    fn from(alg: VectorAlgorithm) -> Self {
        match alg {
            VectorAlgorithm::Linear => EngramDbVectorAlgorithm::Linear,
            VectorAlgorithm::HNSW => EngramDbVectorAlgorithm::HNSW,
            VectorAlgorithm::MultiVector => EngramDbVectorAlgorithm::MultiVector,
        }
    }
}

/// Python enum for multi-vector similarity methods
#[pyclass]
#[derive(Clone, Copy)]
enum MultiVectorSimilarityMethod {
    Maximum = 0,
    Average = 1,
    LateInteraction = 2,
}

impl From<MultiVectorSimilarityMethod> for EngramDbMultiVectorSimilarityMethod {
    fn from(method: MultiVectorSimilarityMethod) -> Self {
        match method {
            MultiVectorSimilarityMethod::Maximum => EngramDbMultiVectorSimilarityMethod::Maximum,
            MultiVectorSimilarityMethod::Average => EngramDbMultiVectorSimilarityMethod::Average,
            MultiVectorSimilarityMethod::LateInteraction => EngramDbMultiVectorSimilarityMethod::LateInteraction,
        }
    }
}

/// Python wrapper for DatabaseConfig
#[pyclass]
struct DatabaseConfig {
    inner: EngramDbDatabaseConfig,
}

#[pymethods]
impl DatabaseConfig {
    /// Create a new in-memory database configuration
    #[staticmethod]
    fn new_in_memory() -> Self {
        Self {
            inner: EngramDbDatabaseConfig::default_in_memory(),
        }
    }
    
    /// Create a new file-based database configuration
    #[staticmethod]
    fn new_file_based(path: &str) -> Self {
        Self {
            inner: EngramDbDatabaseConfig::default_file_based(path),
        }
    }
    
    /// Set the vector algorithm to use
    fn set_vector_algorithm(&mut self, algorithm: VectorAlgorithm) {
        self.inner.set_vector_algorithm(algorithm.into());
    }
    
    /// Set the multi-vector configuration
    fn set_multi_vector_config(&mut self, config: &MultiVectorIndexConfig) {
        self.inner.set_multi_vector_config(config.inner.clone());
    }
}

/// Python wrapper for MultiVectorIndexConfig
#[pyclass]
struct MultiVectorIndexConfig {
    inner: EngramDbMultiVectorIndexConfig,
}

#[pymethods]
impl MultiVectorIndexConfig {
    #[new]
    fn new() -> Self {
        Self {
            inner: EngramDbMultiVectorIndexConfig::default(),
        }
    }
    
    /// Set whether to quantize the vectors
    fn set_quantize(&mut self, quantize: bool) {
        self.inner.quantize = quantize;
    }
    
    /// Set the similarity method to use
    fn set_similarity_method(&mut self, method: MultiVectorSimilarityMethod) {
        self.inner.similarity_method = method.into();
    }
}

/// Python wrapper for MultiVectorEmbedding
#[cfg(feature = "embeddings")]
#[pyclass]
struct MultiVectorEmbedding {
    inner: EngramDbMultiVectorEmbedding,
}

#[cfg(feature = "embeddings")]
#[pymethods]
impl MultiVectorEmbedding {
    /// Returns the dimensions of each vector
    fn dimensions(&self) -> usize {
        self.inner.dimensions()
    }
    
    /// Returns the number of vectors
    fn num_vectors(&self) -> usize {
        self.inner.num_vectors()
    }
    
    /// Returns whether the vectors are quantized
    fn is_quantized(&self) -> bool {
        self.inner.is_quantized()
    }
    
    /// Returns the vectors as a list of numpy arrays
    fn vectors<'py>(&self, py: Python<'py>) -> Vec<&'py PyArray1<f32>> {
        self.inner.vectors().iter()
            .map(|v| v.to_vec().into_pyarray(py))
            .collect()
    }
    
    /// Compute the maximum similarity between any pair of vectors in the two multi-vector embeddings
    fn max_similarity(&self, other: &MultiVectorEmbedding) -> f32 {
        self.inner.max_similarity(&other.inner)
    }
    
    /// Compute the average similarity between all pairs of vectors in the two multi-vector embeddings
    fn avg_similarity(&self, other: &MultiVectorEmbedding) -> f32 {
        self.inner.avg_similarity(&other.inner)
    }
    
    /// Calculate the late interaction score between this and another multi-vector embedding
    fn late_interaction_score(&self, other: &MultiVectorEmbedding) -> f32 {
        self.inner.late_interaction_score(&other.inner)
    }
}

/// Register the Python module
#[pymodule]
fn _engramdb(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sum_as_string, m)?)?;
    m.add_function(wrap_pyfunction!(load_sample_dataset, m)?)?;
    m.add_function(wrap_pyfunction!(load_minimal_dataset, m)?)?;
    m.add_class::<MemoryNode>()?;
    m.add_class::<Database>()?;
    m.add_class::<ThreadSafeDatabase>()?;
    m.add_class::<ThreadSafeDatabasePool>()?;
    m.add_class::<RelationshipType>()?;
    
    // Add multi-vector support
    m.add_class::<DatabaseConfig>()?;
    m.add_class::<VectorAlgorithm>()?;
    
    #[cfg(feature = "embeddings")]
    {
        m.add_class::<EmbeddingService>()?;
        m.add_class::<EmbeddingModelType>()?;
        
        // Add multi-vector classes (require embeddings feature)
        m.add_class::<MultiVectorEmbedding>()?;
        m.add_class::<MultiVectorIndexConfig>()?;
        m.add_class::<MultiVectorSimilarityMethod>()?;
    }
    
    Ok(())
}