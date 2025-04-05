/// Computes the cosine similarity between two vectors
///
/// The cosine similarity measures the cosine of the angle between two vectors,
/// resulting in a value between -1 and 1 where 1 means identical, 0 means orthogonal,
/// and -1 means opposite.
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The cosine similarity value between the two vectors, or None if either vector is empty
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.is_empty() || b.is_empty() || a.len() != b.len() {
        return None;
    }

    let dot = dot_product(a, b);
    let norm_a = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return None;
    }

    Some(dot / (norm_a * norm_b))
}

/// Computes the dot product between two vectors
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The dot product value, or 0.0 if the vectors have different lengths
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Computes the Euclidean distance between two vectors
///
/// # Arguments
///
/// * `a` - First vector
/// * `b` - Second vector
///
/// # Returns
///
/// The Euclidean distance between the two vectors, or None if the vectors have different lengths
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() {
        return None;
    }

    let sum_squared_diff = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y) * (x - y))
        .sum::<f32>();

    Some(sum_squared_diff.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        // Identical vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim - 1.0).abs() < 1e-6);

        // Orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!(sim.abs() < 1e-6);

        // Opposite vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b).unwrap();
        assert!((sim + 1.0).abs() < 1e-6);

        // Different lengths
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0];
        assert_eq!(cosine_similarity(&a, &b), None);

        // Zero vector
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), None);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dot = dot_product(&a, &b);
        assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        // Different lengths
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let dot = dot_product(&a, &b);
        assert_eq!(dot, 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        // Same vector
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0];
        let dist = euclidean_distance(&a, &b).unwrap();
        assert!(dist.abs() < 1e-6);

        // Simple case
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = euclidean_distance(&a, &b).unwrap();
        assert!((dist - 5.196_152).abs() < 1e-6); // sqrt((4-1)^2 + (5-2)^2 + (6-3)^2) = sqrt(27) ≈ 5.196

        // Different lengths
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        assert_eq!(euclidean_distance(&a, &b), None);
    }
}
