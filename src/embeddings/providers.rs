//! Embedding provider implementations
//!
//! This module contains different embedding provider implementations,
//! such as model-based providers and mock providers.

use std::error::Error;
pub use super::mock::MockProvider;
use super::models::EmbeddingModel;

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
                kwargs.set_item("max_length", 512)?;
                
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