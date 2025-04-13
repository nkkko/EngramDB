//! Mock embedding generation for testing and fallback
//! 
//! This module provides functions to generate random embeddings
//! when a real embedding model is not available.

use rand::{thread_rng, Rng, SeedableRng};
use super::core;

/// Generate a random embedding vector with the specified dimensions
pub fn generate_random_embedding(dimensions: usize) -> Vec<f32> {
    let mut rng = thread_rng();
    let random_values: Vec<f32> = (0..dimensions)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // Normalize to unit length
    core::normalize_vector(&random_values)
}

/// Generate a deterministic embedding based on a hash of the input text
pub fn generate_deterministic_embedding(text: &str, dimensions: usize) -> Vec<f32> {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    
    // Hash the input text
    let mut hasher = DefaultHasher::new();
    text.hash(&mut hasher);
    let hash = hasher.finish();
    
    // Use the hash to seed the random number generator
    let mut rng = rand::rngs::StdRng::seed_from_u64(hash);
    
    // Generate deterministic values
    let values: Vec<f32> = (0..dimensions)
        .map(|_| rng.gen_range(-1.0..1.0))
        .collect();
    
    // Normalize
    core::normalize_vector(&values)
}

/// MockProvider generates random or deterministic embeddings
pub struct MockProvider {
    dimensions: usize,
}

impl MockProvider {
    /// Create a new mock provider with specified dimensions
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl super::Provider for MockProvider {
    fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    fn generate(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        Ok(generate_deterministic_embedding(text, self.dimensions))
    }
    
    fn generate_for_document(&self, text: &str, category: Option<&str>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Include category in the seed if provided
        let text_with_category = match category {
            Some(cat) => format!("{}: {}", cat, text),
            None => text.to_string(),
        };
        
        Ok(generate_deterministic_embedding(&text_with_category, self.dimensions))
    }
    
    fn generate_for_query(&self, text: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        // Add a small bias for query embeddings to differentiate them
        let query_text = format!("query: {}", text);
        Ok(generate_deterministic_embedding(&query_text, self.dimensions))
    }
    
    fn generate_random(&self) -> Vec<f32> {
        generate_random_embedding(self.dimensions)
    }
}