// Multi-vector support for Python bindings
// This file should be manually merged into lib.rs

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray1};
use engramdb::{
    DatabaseConfig as EngramDbDatabaseConfig,
    StorageType as EngramDbStorageType,
    EmbeddingModel as EngramDbEmbeddingModel,
    embeddings::multi_vector::{
        MultiVectorEmbedding as EngramDbMultiVectorEmbedding,
        MultiVectorProvider as EngramDbMultiVectorProvider,
    },
    vector::{
        MultiVectorIndexConfig as EngramDbMultiVectorIndexConfig,
        MultiVectorSimilarityMethod as EngramDbMultiVectorSimilarityMethod,
        VectorAlgorithm as EngramDbVectorAlgorithm,
    },
};

// Add to imports in lib.rs

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

/// Python wrapper for MultiVectorEmbedding
#[pyclass]
struct MultiVectorEmbedding {
    inner: EngramDbMultiVectorEmbedding,
}

/// Python wrapper for MultiVectorIndexConfig
#[pyclass]
struct MultiVectorIndexConfig {
    inner: EngramDbMultiVectorIndexConfig,
}

/// Python wrapper for DatabaseConfig
#[pyclass]
struct DatabaseConfig {
    inner: EngramDbDatabaseConfig,
}

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
        self.inner.vector_index_config.algorithm = algorithm.into();
    }
    
    /// Set the multi-vector configuration
    fn set_multi_vector_config(&mut self, config: &MultiVectorIndexConfig) {
        self.inner.vector_index_config.multi_vector = Some(config.inner.clone());
    }
}

// Add to the MemoryNode implementation
impl MemoryNode {
    #[staticmethod]
    fn with_multi_vector(multi_vector: &MultiVectorEmbedding) -> Self {
        Self {
            inner: EngramDbMemoryNode::with_multi_vector(multi_vector.inner.clone()),
        }
    }
    
    #[staticmethod]
    fn from_text_multi_vector(text: &str, embedding_service: &EmbeddingService, category: Option<&str>) -> PyResult<Self> {
        match EngramDbMemoryNode::from_text_multi_vector(
            text, 
            &embedding_service.inner as &dyn EngramDbMultiVectorProvider, 
            category
        ) {
            Ok(node) => Ok(Self { inner: node }),
            Err(e) => Err(PyValueError::new_err(format!("Failed to create memory with multi-vector: {}", e))),
        }
    }
    
    // Check if using multi-vector embeddings
    fn is_multi_vector(&self) -> bool {
        self.inner.is_multi_vector()
    }
    
    // Get multi-vector embeddings if available
    fn multi_vector_embeddings(&self) -> Option<MultiVectorEmbedding> {
        self.inner.multi_vector_embeddings().map(|mv| MultiVectorEmbedding { inner: mv.clone() })
    }
}

// Add to the EmbeddingService implementation
impl EmbeddingService {
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

// Update EmbeddingModelType enum
#[pyclass]
enum EmbeddingModelType {
    E5 = 0,
    GTE = 1,
    JINA = 2,
    JINA_COLBERT = 3,
    CUSTOM = 4,
}

impl From<EmbeddingModelType> for EngramDbEmbeddingModel {
    fn from(model_type: EmbeddingModelType) -> Self {
        match model_type {
            EmbeddingModelType::E5 => EngramDbEmbeddingModel::E5MultilingualLargeInstruct,
            EmbeddingModelType::GTE => EngramDbEmbeddingModel::GteModernBertBase,
            EmbeddingModelType::JINA => EngramDbEmbeddingModel::JinaEmbeddingsV3,
            EmbeddingModelType::JINA_COLBERT => EngramDbEmbeddingModel::JinaColBERTv2,
            EmbeddingModelType::CUSTOM => EngramDbEmbeddingModel::Custom,
        }
    }
}

// Add to the pymodule function
fn register_multi_vector_classes(m: &PyModule) -> PyResult<()> {
    m.add_class::<DatabaseConfig>()?;
    m.add_class::<VectorAlgorithm>()?;
    m.add_class::<MultiVectorSimilarityMethod>()?;
    
    #[cfg(feature = "embeddings")]
    {
        m.add_class::<MultiVectorEmbedding>()?;
        m.add_class::<MultiVectorIndexConfig>()?;
    }
    
    Ok(())
}