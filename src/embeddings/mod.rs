//! Embedding generation module for EngramDB
//!
//! This module provides functionality for generating vector embeddings from text,
//! which can be used for semantic search and memory creation within EngramDB.

mod providers;
// Provider implementations will be re-exported below

/// Core embedding functionality
pub mod core;
pub use core::*;

/// Mock embeddings for testing and fallback
pub mod mock;
pub use mock::*;

/// Model information and supported models
pub mod models;
pub use models::*;

/// Multi-vector embeddings (ColBERT/ColPali-style)
pub mod multi_vector;
pub use multi_vector::{
    MultiVectorEmbedding, MultiVectorProvider, MultiVectorError,
    MockMultiVectorProvider,
};

#[cfg(test)]
mod tests;

// Re-export the Provider trait and implementations
pub use providers::Provider;
pub use providers::ModelProvider;
pub use providers::MockProvider;

// By default, only expose the high-level API
pub use embedding_service::EmbeddingService;
pub use embedding_service::EmbeddingResult;
pub use embedding_service::EmbeddingError;

/// Main embedding service for EngramDB
pub mod embedding_service {
    use super::*;
    use std::sync::{Arc, Mutex, MutexGuard};
    use std::fmt;

    /// Errors that can occur when generating embeddings
    #[derive(Debug, Clone)]
    pub enum EmbeddingError {
        /// Provider is not available or failed to initialize
        ProviderUnavailable(String),
        /// Error generating embeddings
        GenerationFailed(String),
        /// Invalid input provided
        InvalidInput(String),
        /// Multi-vector specific error
        MultiVectorError(String),
    }

    impl fmt::Display for EmbeddingError {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                EmbeddingError::ProviderUnavailable(msg) => write!(f, "Embedding provider unavailable: {}", msg),
                EmbeddingError::GenerationFailed(msg) => write!(f, "Failed to generate embeddings: {}", msg),
                EmbeddingError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
                EmbeddingError::MultiVectorError(msg) => write!(f, "Multi-vector error: {}", msg),
            }
        }
    }

    impl std::error::Error for EmbeddingError {}

    /// Result type for embedding operations
    pub type EmbeddingResult<T> = Result<T, EmbeddingError>;

    /// Main service for generating embeddings
    #[derive(Clone)]
    pub struct EmbeddingService {
        provider: Arc<Mutex<Box<dyn Provider + Send>>>,
        dimensions: usize,
        
        // Optional multi-vector provider
        #[cfg(feature = "embeddings")]
        multi_vector_provider: Option<Arc<Mutex<Box<dyn MultiVectorProvider + Send>>>>,
    }

    impl EmbeddingService {
        /// Creates a new embedding service using the specified provider
        pub fn new<P: Provider + Send + 'static>(provider: P) -> Self {
            let dimensions = provider.dimensions();
            Self {
                provider: Arc::new(Mutex::new(Box::new(provider))),
                dimensions,
                #[cfg(feature = "embeddings")]
                multi_vector_provider: None,
            }
        }

        /// Creates a new embedding service with a mock provider
        pub fn new_mock(dimensions: usize) -> Self {
            Self::new(mock::MockProvider::new(dimensions))
        }
        
        /// Creates a new embedding service with both single-vector and multi-vector providers
        #[cfg(feature = "embeddings")]
        pub fn new_with_multi_vector<P, M>(provider: P, multi_provider: M) -> Self
        where
            P: Provider + Send + 'static,
            M: MultiVectorProvider + Send + 'static,
        {
            let dimensions = provider.dimensions();
            Self {
                provider: Arc::new(Mutex::new(Box::new(provider))),
                dimensions,
                multi_vector_provider: Some(Arc::new(Mutex::new(Box::new(multi_provider)))),
            }
        }
        
        /// Creates a new embedding service with mock providers for both single and multi-vector
        #[cfg(feature = "embeddings")]
        pub fn new_mock_multi_vector(dimensions: usize, num_vectors: usize) -> Self {
            let provider = mock::MockProvider::new(dimensions);
            let multi_provider = multi_vector::MockMultiVectorProvider::new(dimensions, num_vectors);
            Self::new_with_multi_vector(provider, multi_provider)
        }

        /// Creates a new embedding service using the model provider
        /// Will fall back to mock provider if model initialization fails
        pub fn new_with_model(model_name: Option<&str>) -> Self {
            // Default to E5 if no model is specified
            let model_name = model_name.unwrap_or(models::EmbeddingModel::default().model_id());
            
            // Get model type for dimensions
            let model_type = models::EmbeddingModel::from_name(model_name);
            let fallback_dimensions = model_type.dimensions();

            // Try to initialize the model provider
            match providers::ModelProvider::new(model_name) {
                Ok(provider) => {
                    let dimensions = provider.dimensions();
                    Self {
                        provider: Arc::new(Mutex::new(Box::new(provider))),
                        dimensions,
                        #[cfg(feature = "embeddings")]
                        multi_vector_provider: None,
                    }
                },
                Err(err) => {
                    log::warn!("Failed to initialize model provider: {}. Falling back to mock provider.", err);
                    // Fall back to a mock provider with the appropriate dimensions
                    Self::new_mock(fallback_dimensions)
                }
            }
        }
        
        /// Creates a new embedding service using a specific pre-defined model
        pub fn with_model_type(model_type: models::EmbeddingModel) -> Self {
            Self::new_with_model(Some(model_type.model_id()))
        }
        
        /// Creates a new embedding service using a multi-vector model (ColBERT/ColPali-style)
        #[cfg(feature = "embeddings")]
        pub fn with_multi_vector_model(model_type: models::EmbeddingModel) -> Self {
            if !model_type.is_multi_vector() {
                log::warn!("Model {} is not a multi-vector model. Using standard model provider.", model_type.model_id());
                return Self::with_model_type(model_type);
            }
            
            // Create a regular model provider with the ColBERT model
            match providers::ModelProvider::new(model_type.model_id()) {
                Ok(provider) => {
                    let dimensions = provider.dimensions();
                    
                    // The provider will automatically handle multi-vector processing based on the model type
                    let service = Self {
                        provider: Arc::new(Mutex::new(Box::new(provider.clone()))), 
                        dimensions,
                        multi_vector_provider: Some(Arc::new(Mutex::new(Box::new(provider)))),
                    };
                    
                    service
                },
                Err(err) => {
                    log::warn!("Failed to initialize multi-vector provider: {}. Falling back to mock provider.", err);
                    // Fall back to a mock provider
                    Self::new_mock_multi_vector(model_type.dimensions(), model_type.typical_vector_count())
                }
            }
        }
        
        /// Adds multi-vector capability to an existing embedding service
        #[cfg(feature = "embeddings")]
        pub fn add_multi_vector_provider<M>(&mut self, multi_provider: M) 
        where 
            M: MultiVectorProvider + Send + 'static 
        {
            self.multi_vector_provider = Some(Arc::new(Mutex::new(Box::new(multi_provider))));
        }
        
        /// Checks if this service has multi-vector capability
        #[cfg(feature = "embeddings")]
        pub fn has_multi_vector(&self) -> bool {
            self.multi_vector_provider.is_some()
        }
        
        #[cfg(not(feature = "embeddings"))]
        pub fn has_multi_vector(&self) -> bool {
            false
        }

        /// Returns the dimensions of the embeddings
        pub fn dimensions(&self) -> usize {
            self.dimensions
        }
        
        /// Returns the dimensions of the multi-vector embeddings
        #[cfg(feature = "embeddings")]
        pub fn multi_vector_dimensions(&self) -> Option<usize> {
            if let Some(ref provider) = self.multi_vector_provider {
                if let Ok(provider) = provider.lock() {
                    return Some(provider.multi_vector_dimensions());
                }
            }
            None
        }

        /// Generate embeddings for the given text
        pub fn generate(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
            let provider = self.get_provider()?;
            provider.generate(text)
                .map_err(|e| EmbeddingError::GenerationFailed(e.to_string()))
        }

        /// Generate embeddings for a document
        pub fn generate_for_document(&self, text: &str, category: Option<&str>) -> EmbeddingResult<Vec<f32>> {
            let provider = self.get_provider()?;
            provider.generate_for_document(text, category)
                .map_err(|e| EmbeddingError::GenerationFailed(e.to_string()))
        }

        /// Generate embeddings for a query
        pub fn generate_for_query(&self, text: &str) -> EmbeddingResult<Vec<f32>> {
            let provider = self.get_provider()?;
            provider.generate_for_query(text)
                .map_err(|e| EmbeddingError::GenerationFailed(e.to_string()))
        }
        
        /// Generate multi-vector embeddings for the given text
        #[cfg(feature = "embeddings")]
        pub fn generate_multi_vector(&self, text: &str) -> EmbeddingResult<MultiVectorEmbedding> {
            let provider = self.get_multi_vector_provider()?;
            provider.generate_multi_vector(text)
                .map_err(|e| EmbeddingError::GenerationFailed(e.to_string()))
        }
        
        /// Generate multi-vector embeddings for a document
        #[cfg(feature = "embeddings")]
        pub fn generate_multi_vector_for_document(&self, text: &str, category: Option<&str>) 
            -> EmbeddingResult<MultiVectorEmbedding> 
        {
            let provider = self.get_multi_vector_provider()?;
            provider.generate_multi_vector_for_document(text, category)
                .map_err(|e| EmbeddingError::GenerationFailed(e.to_string()))
        }
        
        /// Generate multi-vector embeddings for a query
        #[cfg(feature = "embeddings")]
        pub fn generate_multi_vector_for_query(&self, text: &str) 
            -> EmbeddingResult<MultiVectorEmbedding> 
        {
            let provider = self.get_multi_vector_provider()?;
            provider.generate_multi_vector_for_query(text)
                .map_err(|e| EmbeddingError::GenerationFailed(e.to_string()))
        }

        /// Generate random embeddings with the same dimensions as the provider
        pub fn generate_random(&self) -> Vec<f32> {
            let provider = match self.get_provider() {
                Ok(provider) => provider,
                Err(_) => return mock::generate_random_embedding(self.dimensions),
            };
            
            provider.generate_random()
        }
        
        /// Generate random multi-vector embeddings
        #[cfg(feature = "embeddings")]
        pub fn generate_random_multi_vector(&self, num_vectors: Option<usize>) 
            -> Option<MultiVectorEmbedding> 
        {
            let provider = match self.get_multi_vector_provider() {
                Ok(provider) => provider,
                Err(_) => return None,
            };
            
            Some(provider.generate_random_multi_vector(num_vectors))
        }

        // Get a lock on the provider
        fn get_provider(&self) -> EmbeddingResult<MutexGuard<Box<dyn Provider + Send>>> {
            self.provider.lock()
                .map_err(|e| EmbeddingError::ProviderUnavailable(format!("Failed to acquire lock: {}", e)))
        }
        
        // Get a lock on the multi-vector provider
        #[cfg(feature = "embeddings")]
        fn get_multi_vector_provider(&self) -> EmbeddingResult<MutexGuard<Box<dyn MultiVectorProvider + Send>>> {
            if let Some(ref provider) = self.multi_vector_provider {
                provider.lock()
                    .map_err(|e| EmbeddingError::ProviderUnavailable(format!("Failed to acquire lock: {}", e)))
            } else {
                Err(EmbeddingError::ProviderUnavailable("Multi-vector provider not available".to_string()))
            }
        }
    }

    impl Default for EmbeddingService {
        fn default() -> Self {
            // Default to mock provider for safety
            Self::new_mock(384)
        }
    }
}