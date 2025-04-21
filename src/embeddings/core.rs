//! Core embedding functionality and utilities

/// Normalizes a vector to unit length (L2 norm)
pub fn normalize_vector(vector: &[f32]) -> Vec<f32> {
    let sum_squares: f32 = vector.iter().map(|&x| x * x).sum();
    let norm = sum_squares.sqrt();

    if norm > 0.0 {
        vector.iter().map(|&x| x / norm).collect()
    } else {
        // Return the original vector if norm is zero (shouldn't happen with real embeddings)
        vector.to_vec()
    }
}

/// Calculates the cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();

    let a_norm_sq: f32 = a.iter().map(|&x| x * x).sum();
    let b_norm_sq: f32 = b.iter().map(|&x| x * x).sum();

    let a_norm = a_norm_sq.sqrt();
    let b_norm = b_norm_sq.sqrt();

    if a_norm > 0.0 && b_norm > 0.0 {
        dot_product / (a_norm * b_norm)
    } else {
        0.0
    }
}

/// Calculates the dot product between two vectors
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Adds two vectors element-wise
pub fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    if a.len() != b.len() {
        return Vec::new();
    }

    a.iter().zip(b).map(|(x, y)| x + y).collect()
}
