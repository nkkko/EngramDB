//! Multi-vector embedding functionality
//!
//! This module provides support for ColBERT/ColPali-style multi-vector embeddings,
//! which represent documents as multiple vectors rather than a single vector.

use std::error::Error;
use super::core::{cosine_similarity, normalize_vector};

/// Represents a document as a collection of vectors (multi-vector embedding)
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct MultiVectorEmbedding {
    /// Collection of vectors representing different aspects of the document
    /// Each inner vector typically has dimensions in the range of 96-128
    vectors: Vec<Vec<f32>>,
    
    /// Original dimensionality of each vector
    dimensions: usize,
    
    /// Whether the vectors are quantized for memory efficiency
    is_quantized: bool,
}

/// Errors specific to multi-vector operations
#[derive(Debug, thiserror::Error)]
pub enum MultiVectorError {
    #[error("Empty multi-vector embedding")]
    EmptyEmbedding,
    
    #[error("Dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    
    #[error("Invalid multi-vector format: {0}")]
    InvalidFormat(String),
    
    #[error("Quantization error: {0}")]
    QuantizationError(String),
}

impl MultiVectorEmbedding {
    /// Create a new multi-vector embedding from a collection of vectors
    pub fn new(vectors: Vec<Vec<f32>>) -> Result<Self, MultiVectorError> {
        if vectors.is_empty() {
            return Err(MultiVectorError::EmptyEmbedding);
        }
        
        // Check that all vectors have the same dimensionality
        let dimensions = vectors[0].len();
        for (_i, vec) in vectors.iter().enumerate().skip(1) {
            if vec.len() != dimensions {
                return Err(MultiVectorError::DimensionMismatch { 
                    expected: dimensions, 
                    actual: vec.len() 
                });
            }
        }
        
        Ok(Self {
            vectors,
            dimensions,
            is_quantized: false,
        })
    }
    
    /// Create a new multi-vector embedding with normalized vectors
    pub fn new_normalized(vectors: Vec<Vec<f32>>) -> Result<Self, MultiVectorError> {
        if vectors.is_empty() {
            return Err(MultiVectorError::EmptyEmbedding);
        }
        
        // Normalize each vector
        let normalized_vectors = vectors
            .iter()
            .map(|v| normalize_vector(v))
            .collect();
            
        Self::new(normalized_vectors)
    }
    
    /// Convert a single vector embedding into a multi-vector with a single element
    pub fn from_single_vector(vector: Vec<f32>) -> Result<Self, MultiVectorError> {
        Self::new(vec![vector])
    }
    
    /// Apply quantization to reduce memory footprint (8-bit quantization)
    pub fn quantize(&mut self) -> Result<(), MultiVectorError> {
        // Implementation of quantization would go here
        // For now, we're just marking it as quantized
        self.is_quantized = true;
        Ok(())
    }
    
    /// Returns the inner vectors
    pub fn vectors(&self) -> &[Vec<f32>] {
        &self.vectors
    }
    
    /// Returns the dimensionality of each vector
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
    
    /// Returns the number of vectors
    pub fn num_vectors(&self) -> usize {
        self.vectors.len()
    }
    
    /// Returns whether the vectors are quantized
    pub fn is_quantized(&self) -> bool {
        self.is_quantized
    }
    
    /// Compute the maximum similarity between any pair of vectors in the two multi-vector embeddings
    pub fn max_similarity(&self, other: &MultiVectorEmbedding) -> f32 {
        let mut max_sim = 0.0f32;
        
        for a in &self.vectors {
            for b in &other.vectors {
                let sim = cosine_similarity(a, b);
                max_sim = max_sim.max(sim);
            }
        }
        
        max_sim
    }
    
    /// Compute the average similarity between all pairs of vectors in the two multi-vector embeddings
    pub fn avg_similarity(&self, other: &MultiVectorEmbedding) -> f32 {
        let mut total_sim = 0.0f32;
        let mut count = 0;
        
        for a in &self.vectors {
            for b in &other.vectors {
                total_sim += cosine_similarity(a, b);
                count += 1;
            }
        }
        
        if count > 0 {
            total_sim / (count as f32)
        } else {
            0.0
        }
    }
    
    /// Calculate the late interaction score between this and another multi-vector embedding
    /// This is based on the ColBERT-style maxsim operation
    pub fn late_interaction_score(&self, other: &MultiVectorEmbedding) -> f32 {
        let mut total_score = 0.0f32;
        
        // For each vector in self, find the max similarity with any vector in other
        for a in &self.vectors {
            let mut max_sim = 0.0f32;
            for b in &other.vectors {
                let sim = cosine_similarity(a, b);
                max_sim = max_sim.max(sim);
            }
            total_score += max_sim;
        }
        
        // Normalize by the number of vectors in self
        if !self.vectors.is_empty() {
            total_score / (self.vectors.len() as f32)
        } else {
            0.0
        }
    }
}

/// Provider trait extension for generating multi-vector embeddings
pub trait MultiVectorProvider {
    /// Returns the dimensions of each vector in the multi-vector embeddings
    fn multi_vector_dimensions(&self) -> usize;
    
    /// Returns the typical number of vectors per document
    fn typical_vector_count(&self) -> usize;
    
    /// Generate multi-vector embeddings for the given text
    fn generate_multi_vector(&self, text: &str) -> Result<MultiVectorEmbedding, Box<dyn Error>>;
    
    /// Generate multi-vector embeddings for a document to be stored
    fn generate_multi_vector_for_document(&self, text: &str, category: Option<&str>) -> Result<MultiVectorEmbedding, Box<dyn Error>>;
    
    /// Generate multi-vector embeddings for a query (for similarity search)
    fn generate_multi_vector_for_query(&self, text: &str) -> Result<MultiVectorEmbedding, Box<dyn Error>>;
    
    /// Generate random multi-vector embeddings
    fn generate_random_multi_vector(&self, num_vectors: Option<usize>) -> MultiVectorEmbedding;
}

/// Mock implementation of MultiVectorProvider for testing
pub struct MockMultiVectorProvider {
    dimensions: usize,
    typical_count: usize,
}

impl MockMultiVectorProvider {
    pub fn new(dimensions: usize, typical_count: usize) -> Self {
        Self { dimensions, typical_count }
    }
    
    fn generate_random_vectors(&self, count: usize) -> Vec<Vec<f32>> {
        let mut vectors = Vec::with_capacity(count);
        for _ in 0..count {
            vectors.push(super::mock::generate_random_embedding(self.dimensions));
        }
        vectors
    }
}

impl MultiVectorProvider for MockMultiVectorProvider {
    fn multi_vector_dimensions(&self) -> usize {
        self.dimensions
    }
    
    fn typical_vector_count(&self) -> usize {
        self.typical_count
    }
    
    fn generate_multi_vector(&self, _text: &str) -> Result<MultiVectorEmbedding, Box<dyn Error>> {
        let vectors = self.generate_random_vectors(self.typical_count);
        Ok(MultiVectorEmbedding::new(vectors)?)
    }
    
    fn generate_multi_vector_for_document(&self, text: &str, _category: Option<&str>) -> Result<MultiVectorEmbedding, Box<dyn Error>> {
        // Use the text length to determine a variable number of vectors
        let length = text.len();
        let count = std::cmp::max(1, std::cmp::min(length / 20, self.typical_count * 2));
        
        let vectors = self.generate_random_vectors(count);
        Ok(MultiVectorEmbedding::new(vectors)?)
    }
    
    fn generate_multi_vector_for_query(&self, _text: &str) -> Result<MultiVectorEmbedding, Box<dyn Error>> {
        // Queries typically have fewer vectors than documents
        let count = self.typical_count / 2;
        let vectors = self.generate_random_vectors(count);
        Ok(MultiVectorEmbedding::new(vectors)?)
    }
    
    fn generate_random_multi_vector(&self, num_vectors: Option<usize>) -> MultiVectorEmbedding {
        let count = num_vectors.unwrap_or(self.typical_count);
        let vectors = self.generate_random_vectors(count);
        MultiVectorEmbedding::new(vectors).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_multi_vector_creation() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let multi_vec = MultiVectorEmbedding::new(vectors.clone()).unwrap();
        
        assert_eq!(multi_vec.dimensions(), 3);
        assert_eq!(multi_vec.num_vectors(), 3);
        assert_eq!(multi_vec.vectors(), vectors.as_slice());
        assert!(!multi_vec.is_quantized());
    }
    
    #[test]
    fn test_multi_vector_similarity() {
        let vectors1 = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        
        let vectors2 = vec![
            vec![0.8, 0.2, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let multi_vec1 = MultiVectorEmbedding::new(vectors1).unwrap();
        let multi_vec2 = MultiVectorEmbedding::new(vectors2).unwrap();
        
        let max_sim = multi_vec1.max_similarity(&multi_vec2);
        let avg_sim = multi_vec1.avg_similarity(&multi_vec2);
        let late_sim = multi_vec1.late_interaction_score(&multi_vec2);
        
        // First vector in multi_vec1 has high similarity with first vector in multi_vec2
        assert!(max_sim > 0.5);
        // Average should be lower since not all vectors are similar
        assert!(avg_sim < max_sim);
        // Late interaction score should be in between
        assert!(late_sim >= avg_sim && late_sim <= max_sim);
    }
    
    #[test]
    fn test_mock_provider() {
        let provider = MockMultiVectorProvider::new(96, 20);
        
        let multi_vec = provider.generate_random_multi_vector(None);
        assert_eq!(multi_vec.dimensions(), 96);
        assert_eq!(multi_vec.num_vectors(), 20);
        
        let multi_vec = provider.generate_random_multi_vector(Some(10));
        assert_eq!(multi_vec.dimensions(), 96);
        assert_eq!(multi_vec.num_vectors(), 10);
        
        let result = provider.generate_multi_vector("Test text").unwrap();
        assert_eq!(result.dimensions(), 96);
        assert_eq!(result.num_vectors(), 20);
        
        let result = provider.generate_multi_vector_for_query("Query text").unwrap();
        assert_eq!(result.dimensions(), 96);
        assert_eq!(result.num_vectors(), 10); // Half of typical count
    }
    
    #[test]
    fn test_dimension_mismatch() {
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0], // Different length
        ];
        
        let result = MultiVectorEmbedding::new(vectors);
        assert!(result.is_err());
        
        if let Err(MultiVectorError::DimensionMismatch { expected, actual }) = result {
            assert_eq!(expected, 3);
            assert_eq!(actual, 2);
        } else {
            panic!("Expected DimensionMismatch error");
        }
    }
}