//! Embedding provider implementations
//!
//! This module contains different embedding provider implementations,
//! such as model-based providers and mock providers.

use std::error::Error;
pub use super::mock::MockProvider;
use super::models::EmbeddingModel;
use super::multi_vector::{MultiVectorEmbedding, MultiVectorProvider};

/// Trait for embedding providers
pub trait Provider {
    /// Returns the dimensions of the embeddings
    fn dimensions(&self) -> usize;
    
    /// Generate embeddings for the given text
    fn generate(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>>;
    
    /// Generate embeddings for a document to be stored
    fn generate_for_document(&self, text: &str, category: Option<&str>) -> Result<Vec<f32>, Box<dyn Error>>;
    
    /// Generate embeddings for a query (for similarity search)
    fn generate_for_query(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>>;
    
    /// Generate random embeddings with the same dimensions
    fn generate_random(&self) -> Vec<f32>;
}

/// Provider that uses an ML model for generating embeddings
#[derive(Clone)]
pub struct ModelProvider {
    dimensions: usize,
    /// Model type being used
    #[allow(dead_code)]
    model_type: EmbeddingModel,
    /// This field will hold PyO3 model references when Python is available
    #[allow(dead_code)]
    model_data: Option<ModelData>,
}

/// Contains Python model objects and related data
#[allow(dead_code)]
#[derive(Clone)]
enum ModelData {
    /// When compiled with PyO3 feature
    #[cfg(feature = "python")]
    Python {
        model: pyo3::PyObject,
        tokenizer: pyo3::PyObject,
    },
    
    /// When not using PyO3
    #[cfg(not(feature = "python"))]
    Empty,
}

impl ModelProvider {
    /// Create a new model provider with the specified model name
    pub fn new(model_name: &str) -> Result<Self, Box<dyn Error>> {
        // Determine the model type from the name
        let model_type = EmbeddingModel::from_name(model_name);
        
        // Get dimensions based on model type
        let _dimensions = model_type.dimensions();
        
        // If it's a custom model, use the actual model name, otherwise use the canonical ID
        let _model_id = if model_type == EmbeddingModel::Custom {
            model_name
        } else {
            model_type.model_id()
        };
        
        #[cfg(feature = "python")]
        {
            use pyo3::prelude::*;
            use pyo3::types::PyDict;
            
            // Initialize Python interpreter
            pyo3::prepare_freethreaded_python();
            
            Python::with_gil(|py| {
                // Import the transformers module
                let transformers = py.import("transformers")?;
                let torch = py.import("torch")?;
                
                // Create tokenizer and model
                let auto_tokenizer = transformers.getattr("AutoTokenizer")?;
                let auto_model = transformers.getattr("AutoModel")?;
                
                // Create kwargs dict for both model and tokenizer
                let kwargs = PyDict::new(py);
                kwargs.set_item("cache_dir", "model_cache")?;
                
                // Load the tokenizer and model
                let tokenizer = auto_tokenizer.call_method("from_pretrained", (model_id,), Some(kwargs))?;
                let model = auto_model.call_method("from_pretrained", (model_id,), Some(kwargs))?;
                
                // Check if CUDA is available and move model to GPU if possible
                let cuda_available = torch.getattr("cuda")?.call_method0("is_available")?.extract::<bool>()?;
                if cuda_available {
                    let cuda = torch.getattr("cuda")?;
                    let device = cuda.call_method0("current_device")?;
                    model.call_method("to", (device,), None)?;
                }
                
                // For custom models, try to get the dimensions from the model's config
                if model_type == EmbeddingModel::Custom {
                    if let Ok(config) = model.getattr("config") {
                        if let Ok(dim) = config.getattr("hidden_size") {
                            if let Ok(dim_val) = dim.extract::<usize>() {
                                dimensions = dim_val;
                                log::info!("Using custom model with dimension {}", dimensions);
                            }
                        }
                    }
                }
                
                Ok(Self {
                    dimensions,
                    model_type,
                    model_data: Some(ModelData::Python {
                        model: model.into(),
                        tokenizer: tokenizer.into(),
                    }),
                })
            })
        }
        
        #[cfg(not(feature = "python"))]
        {
            log::warn!("Python support not enabled. Using mock embedding provider.");
            Err("Python support not enabled. Enable the 'python' feature.".into())
        }
    }
}

impl Provider for ModelProvider {
    fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    fn generate(&self, text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        // Default to document embedding
        self.generate_for_document(text, None)
    }
    
    fn generate_for_document(&self, _text: &str, _category: Option<&str>) -> Result<Vec<f32>, Box<dyn Error>> {
        #[cfg(feature = "python")]
        {
            use pyo3::prelude::*;
            use pyo3::types::{PyDict, PyList};
            
            // Format text with optional category
            let formatted_text = match _category {
                Some(cat) => format!("Represent the {} document for retrieval: {}", cat, _text),
                None => format!("Represent the document for retrieval: {}", _text),
            };
            
            let model_data = match &self.model_data {
                Some(ModelData::Python { model, tokenizer }) => (model, tokenizer),
                _ => return Err("Model not initialized".into()),
            };
            
            Python::with_gil(|py| {
                let (model, tokenizer) = model_data;
                
                // Convert Python objects to references
                let model = model.as_ref(py);
                let tokenizer = tokenizer.as_ref(py);
                
                // Tokenize input
                let kwargs = PyDict::new(py);
                kwargs.set_item("return_tensors", "pt")?;
                kwargs.set_item("padding", true)?;
                kwargs.set_item("truncation", true)?;
                
                // Jina-ColBERT models can handle longer sequences but we use smaller length for efficiency
                let max_length = match self.model_type {
                    crate::embeddings::EmbeddingModel::JinaColBERTv2 => 256,
                    _ => 512,
                };
                kwargs.set_item("max_length", max_length)?;
                
                let inputs = tokenizer.call_method("__call__", (formatted_text,), Some(kwargs))?;
                
                // Move inputs to the same device as the model
                let torch = py.import("torch")?;
                let cuda_available = torch.getattr("cuda")?.call_method0("is_available")?.extract::<bool>()?;
                
                if cuda_available {
                    let device = torch.getattr("cuda")?
                        .call_method0("current_device")?;
                    
                    // We need to iterate inputs dictionary and move each tensor to device
                    for (key, value) in inputs.as_ref(py).downcast::<PyDict>()? {
                        let to_result = value.call_method("to", (device,), None)?;
                        inputs.as_ref(py).downcast::<PyDict>()?.set_item(key, to_result)?;
                    }
                }
                
                // Generate embeddings with no_grad context
                let no_grad = torch.getattr("no_grad")?;
                let ctx_manager = no_grad.call0()?;
                let _ = ctx_manager.call_method0("__enter__")?;
                
                // Call model with **inputs syntax
                let outputs = model.call_method("__call__", (), Some(inputs.as_ref(py).downcast::<PyDict>()?))?;
                
                // Get the last_hidden_state[:, 0, :] for CLS token
                let last_hidden_state = outputs.getattr("last_hidden_state")?;
                let embeddings = last_hidden_state.call_method("index_select", (0, 0), None)?;
                
                // Normalize embeddings
                let functional = torch.getattr("nn")?.getattr("functional")?;
                let normalized = functional.call_method("normalize", (embeddings,), Some(PyDict::new(py).apply(|d| {
                    d.set_item("p", 2).unwrap();
                    d.set_item("dim", 1).unwrap();
                })?))?;
                
                // Convert to numpy and flatten to Rust Vec
                let numpy_embeddings = normalized.call_method0("cpu")?
                    .call_method0("detach")?
                    .call_method0("numpy")?
                    .call_method0("flatten")?;
                
                // Convert numpy array to Vec<f32>
                let py_list = numpy_embeddings.downcast::<PyList>()?;
                let mut result = Vec::with_capacity(py_list.len());
                
                for item in py_list {
                    result.push(item.extract::<f32>()?);
                }
                
                let _ = ctx_manager.call_method0("__exit__")?;
                
                Ok(result)
            })
        }
        
        #[cfg(not(feature = "python"))]
        {
            Err("Python support not enabled".into())
        }
    }
    
    fn generate_for_query(&self, _text: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        #[cfg(feature = "python")]
        {
            // For query, we use a specific instruction
            let formatted_text = format!("query: {}", _text);
            self.generate(&formatted_text)
        }
        
        #[cfg(not(feature = "python"))]
        {
            Err("Python support not enabled".into())
        }
    }
    
    fn generate_random(&self) -> Vec<f32> {
        super::mock::generate_random_embedding(self.dimensions)
    }
}

// Implement multi-vector support for the ModelProvider
impl MultiVectorProvider for ModelProvider {
    fn multi_vector_dimensions(&self) -> usize {
        self.dimensions
    }
    
    fn typical_vector_count(&self) -> usize {
        self.model_type.typical_vector_count()
    }
    
    fn generate_multi_vector(&self, text: &str) -> Result<MultiVectorEmbedding, Box<dyn Error>> {
        // Default to document embedding
        self.generate_multi_vector_for_document(text, None)
    }
    
    fn generate_multi_vector_for_document(&self, _text: &str, _category: Option<&str>) -> Result<MultiVectorEmbedding, Box<dyn Error>> {
        if !self.model_type.is_multi_vector() {
            // This model doesn't support multi-vector directly, create a single-vector wrapper
            let embedding = self.generate_for_document(_text, _category)?;
            return Ok(MultiVectorEmbedding::from_single_vector(embedding)?);
        }
        
        #[cfg(feature = "python")]
        {
            use pyo3::prelude::*;
            use pyo3::types::{PyDict, PyList};
            
            // Format text with optional category for different models
            let formatted_text = match self.model_type {
                crate::embeddings::EmbeddingModel::JinaColBERTv2 => {
                    // For Jina-ColBERT, use specific prefix format
                    match _category {
                        Some(cat) => format!("passage: [{}] {}", cat, _text),
                        None => format!("passage: {}", _text),
                    }
                },
                _ => {
                    // For other models, use standard format
                    match _category {
                        Some(cat) => format!("Represent the {} document for retrieval: {}", cat, _text),
                        None => format!("Represent the document for retrieval: {}", _text),
                    }
                }
            };
            
            let model_data = match &self.model_data {
                Some(ModelData::Python { model, tokenizer }) => (model, tokenizer),
                _ => return Err("Model not initialized".into()),
            };
            
            Python::with_gil(|py| {
                let (model, tokenizer) = model_data;
                
                // Convert Python objects to references
                let model = model.as_ref(py);
                let tokenizer = tokenizer.as_ref(py);
                
                // Tokenize input
                let kwargs = PyDict::new(py);
                kwargs.set_item("return_tensors", "pt")?;
                kwargs.set_item("padding", true)?;
                kwargs.set_item("truncation", true)?;
                
                // Jina-ColBERT models can handle longer sequences but we use smaller length for efficiency
                let max_length = match self.model_type {
                    crate::embeddings::EmbeddingModel::JinaColBERTv2 => 256,
                    _ => 512,
                };
                kwargs.set_item("max_length", max_length)?;
                
                let inputs = tokenizer.call_method("__call__", (formatted_text,), Some(kwargs))?;
                
                // Move inputs to the same device as the model
                let torch = py.import("torch")?;
                let cuda_available = torch.getattr("cuda")?.call_method0("is_available")?.extract::<bool>()?;
                
                if cuda_available {
                    let device = torch.getattr("cuda")?
                        .call_method0("current_device")?;
                    
                    // Move tensors to device
                    for (key, value) in inputs.as_ref(py).downcast::<PyDict>()? {
                        let to_result = value.call_method("to", (device,), None)?;
                        inputs.as_ref(py).downcast::<PyDict>()?.set_item(key, to_result)?;
                    }
                }
                
                // Generate embeddings with no_grad context
                let no_grad = torch.getattr("no_grad")?;
                let ctx_manager = no_grad.call0()?;
                let _ = ctx_manager.call_method0("__enter__")?;
                
                // Call model with **inputs syntax
                let outputs = model.call_method("__call__", (), Some(inputs.as_ref(py).downcast::<PyDict>()?))?;
                
                // For ColBERT models, we use the full last_hidden_state instead of just the CLS token
                let last_hidden_state = outputs.getattr("last_hidden_state")?;
                
                // Get attention mask to identify valid tokens
                let attention_mask = inputs.as_ref(py).downcast::<PyDict>()?.get_item("attention_mask")?;
                
                // Normalize the vectors at each token position
                let functional = torch.getattr("nn")?.getattr("functional")?;
                let normalized = functional.call_method("normalize", (last_hidden_state,), Some(PyDict::new(py).apply(|d| {
                    d.set_item("p", 2).unwrap();
                    d.set_item("dim", 2).unwrap();
                })?))?;
                
                // Move to CPU and convert to numpy for processing
                let cpu_normalized = normalized.call_method0("cpu")?;
                let cpu_mask = attention_mask.call_method0("cpu")?;
                
                // Get the first batch item (we only have one item)
                let batch_normalized = cpu_normalized.call_method("index_select", (0, 0), None)?;
                let batch_mask = cpu_mask.call_method("index_select", (0, 0), None)?;
                
                // Convert to numpy
                let np_normalized = batch_normalized.call_method0("detach")?.call_method0("numpy")?;
                let np_mask = batch_mask.call_method0("detach")?.call_method0("numpy")?;
                
                // Use numpy to only keep the vectors where mask == 1 (valid tokens)
                let numpy = py.import("numpy")?;
                let valid_indices = numpy.call_method1("where", (np_mask.call_method1("__eq__", (1,))?,))?;
                let valid_vectors = np_normalized.call_method("__getitem__", (valid_indices.call_method("__getitem__", (0,))?,), None)?;
                
                // Convert numpy arrays to Rust vectors
                let mut token_vectors = Vec::new();
                
                // Get the number of valid tokens
                let num_tokens = valid_vectors.call_method0("__len__")?
                    .extract::<usize>()?;
                
                // Limit tokens for efficiency if needed
                let max_tokens = if self.model_type == crate::embeddings::EmbeddingModel::JinaColBERTv2 {
                    // For Jina-ColBERT, keep more tokens (up to 64) to improve retrieval quality
                    std::cmp::min(num_tokens, 64)
                } else {
                    // For other models, limit to typical_vector_count
                    std::cmp::min(num_tokens, self.model_type.typical_vector_count())
                };
                
                // Process each token's vector
                for i in 0..max_tokens {
                    // Get the vector for this token
                    let token_vec = valid_vectors.call_method("__getitem__", (i,), None)?;
                    
                    // Convert to list
                    let py_list = token_vec.call_method0("tolist")?;
                    let vector = py_list.extract::<Vec<f32>>()?;
                    
                    token_vectors.push(vector);
                }
                
                let _ = ctx_manager.call_method0("__exit__")?;
                
                // Create the multi-vector embedding
                MultiVectorEmbedding::new(token_vectors)
                    .map_err(|e| e.to_string().into())
            })
        }
        
        #[cfg(not(feature = "python"))]
        {
            Err("Python support not enabled".into())
        }
    }
    
    fn generate_multi_vector_for_query(&self, _text: &str) -> Result<MultiVectorEmbedding, Box<dyn Error>> {
        if !self.model_type.is_multi_vector() {
            // For non-multi-vector models, just wrap the single vector
            let embedding = self.generate_for_query(_text)?;
            return Ok(MultiVectorEmbedding::from_single_vector(embedding)?);
        }
        
        #[cfg(feature = "python")]
        {
            // Format with query prefix based on model type
            let formatted_text = match self.model_type {
                crate::embeddings::EmbeddingModel::JinaColBERTv2 => {
                    // For Jina-ColBERT, use their specific query format
                    format!("query: {}", _text)
                },
                _ => {
                    // For other models, use generic query format
                    format!("search query: {}", _text)
                }
            };
            self.generate_multi_vector(&formatted_text)
        }
        
        #[cfg(not(feature = "python"))]
        {
            Err("Python support not enabled".into())
        }
    }
    
    fn generate_random_multi_vector(&self, num_vectors: Option<usize>) -> MultiVectorEmbedding {
        let count = num_vectors.unwrap_or_else(|| self.typical_vector_count());
        let dimensions = self.dimensions;
        
        let mut vectors = Vec::with_capacity(count);
        for _ in 0..count {
            vectors.push(super::mock::generate_random_embedding(dimensions));
        }
        
        MultiVectorEmbedding::new(vectors).unwrap()
    }
}