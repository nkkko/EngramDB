//! Vector search and similarity operations
//!
//! This module handles vector embedding operations including
//! similarity computations and retrieval by vector similarity.

mod similarity;
mod vector_index;

pub use similarity::{cosine_similarity, dot_product, euclidean_distance};
pub use vector_index::VectorIndex;
