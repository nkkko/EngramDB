#[cfg(test)]
mod tests {
    // No need to import all from super as we're importing specific items below
    use crate::core::MemoryNode;
    use crate::embeddings::multi_vector::MultiVectorEmbedding;
    use crate::EmbeddingService;
    use crate::vector::{
        MultiVectorIndex, MultiVectorIndexConfig, MultiVectorSimilarityMethod,
        VectorAlgorithm, VectorIndexConfig, create_vector_index
    };
    
    #[test]
    fn test_multi_vector_embedding_creation() {
        // Create a multi-vector embedding
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let multi_vec = MultiVectorEmbedding::new(vectors.clone()).unwrap();
        
        assert_eq!(multi_vec.dimensions(), 3);
        assert_eq!(multi_vec.num_vectors(), 3);
        assert_eq!(multi_vec.vectors(), vectors.as_slice());
    }
    
    #[test]
    fn test_memory_node_with_multi_vector() {
        // Create a multi-vector embedding
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];
        
        let multi_vec = MultiVectorEmbedding::new(vectors).unwrap();
        
        // Create a memory node with the multi-vector embedding
        let node = MemoryNode::with_multi_vector(multi_vec.clone());
        
        // Verify the memory node properties
        assert!(node.is_multi_vector());
        assert!(node.embeddings().is_none()); // No single vector
        
        if let Some(retrieved_multi_vec) = node.multi_vector_embeddings() {
            assert_eq!(retrieved_multi_vec.dimensions(), 3);
            assert_eq!(retrieved_multi_vec.num_vectors(), 3);
        } else {
            panic!("Expected multi-vector embeddings");
        }
    }
    
    #[test]
    fn test_embedding_service_with_multi_vector() {
        // Create an embedding service with multi-vector support
        let service = EmbeddingService::new_mock_multi_vector(96, 20);
        
        assert!(service.has_multi_vector());
        assert_eq!(service.dimensions(), 96);
        assert_eq!(service.multi_vector_dimensions(), Some(96));
        
        // Generate multi-vector embeddings
        let text = "This is a test document for multi-vector embeddings.";
        let multi_vec = service.generate_multi_vector_for_document(text, None).unwrap();
        
        assert_eq!(multi_vec.dimensions(), 96);
        assert_eq!(multi_vec.num_vectors(), 20);
        
        // Create a memory node with the multi-vector embedding
        let node = MemoryNode::with_multi_vector(multi_vec);
        
        // Verify the memory node has multi-vector embeddings
        assert!(node.is_multi_vector());
    }
    
    #[test]
    fn test_multi_vector_index() {
        // Create an embedding service with multi-vector support
        let service = EmbeddingService::new_mock_multi_vector(96, 20);
        
        // Create test documents
        let documents = vec![
            "Document about artificial intelligence and machine learning.",
            "Information about natural language processing and transformers.",
            "Research paper on vector embeddings and semantic search.",
            "Tutorial on implementing HNSW algorithm for vector search.",
            "Overview of ColBERT and late interaction mechanisms.",
        ];
        
        // Create memory nodes with multi-vector embeddings
        let mut nodes = Vec::new();
        for doc in &documents {
            let multi_vec = service.generate_multi_vector_for_document(doc, None).unwrap();
            let node = MemoryNode::with_multi_vector(multi_vec);
            nodes.push(node);
        }
        
        // Create a multi-vector index
        let mut index = MultiVectorIndex::new();
        
        // Add nodes to the index
        for node in &nodes {
            index.add(node).unwrap();
        }
        
        assert_eq!(index.len(), nodes.len());
        
        // Test search with different similarity methods
        let multi_vec_query = service.generate_multi_vector_for_query("vector search algorithms").unwrap();
        
        // Test with MaxSim
        let config_max = MultiVectorIndexConfig {
            quantize: false,
            similarity_method: MultiVectorSimilarityMethod::Maximum,
        };
        let mut index_max = MultiVectorIndex::with_config(config_max);
        for node in &nodes {
            index_max.add(node).unwrap();
        }
        let results_max = index_max.search(&multi_vec_query, 3, 0.0).unwrap();
        assert_eq!(results_max.len(), 3);
        
        // Test with AvgSim
        let config_avg = MultiVectorIndexConfig {
            quantize: false,
            similarity_method: MultiVectorSimilarityMethod::Average,
        };
        let mut index_avg = MultiVectorIndex::with_config(config_avg);
        for node in &nodes {
            index_avg.add(node).unwrap();
        }
        let results_avg = index_avg.search(&multi_vec_query, 3, 0.0).unwrap();
        assert_eq!(results_avg.len(), 3);
        
        // Test with LateInteraction
        let config_late = MultiVectorIndexConfig {
            quantize: false,
            similarity_method: MultiVectorSimilarityMethod::LateInteraction,
        };
        let mut index_late = MultiVectorIndex::with_config(config_late);
        for node in &nodes {
            index_late.add(node).unwrap();
        }
        let results_late = index_late.search(&multi_vec_query, 3, 0.0).unwrap();
        assert_eq!(results_late.len(), 3);
    }
    
    #[test]
    fn test_multi_vector_factory() {
        // Create an embedding service with multi-vector support
        let service = EmbeddingService::new_mock_multi_vector(96, 20);
        
        // Create test documents
        let documents = vec![
            "Document about artificial intelligence and machine learning.",
            "Information about natural language processing and transformers.",
            "Research paper on vector embeddings and semantic search.",
        ];
        
        // Create memory nodes with multi-vector embeddings
        let mut nodes = Vec::new();
        for doc in &documents {
            let multi_vec = service.generate_multi_vector_for_document(doc, None).unwrap();
            let node = MemoryNode::with_multi_vector(multi_vec);
            nodes.push(node);
        }
        
        // Create a vector index config for multi-vector
        let config = VectorIndexConfig {
            algorithm: VectorAlgorithm::MultiVector,
            hnsw: None,
            multi_vector: Some(MultiVectorIndexConfig::default()),
        };
        
        // Create a vector index using the factory
        let mut index = create_vector_index(&config);
        
        // Add nodes to the index
        for node in &nodes {
            index.add(node).unwrap();
        }
        
        assert_eq!(index.len(), nodes.len());
        
        // Test search
        let multi_vec_query = service.generate_multi_vector_for_query("vector search").unwrap();
        
        // Convert to regular vector for the common interface
        let query_vec = multi_vec_query.vectors().first().unwrap().clone();
        let results = index.search(&query_vec, 2, 0.0).unwrap();
        
        assert_eq!(results.len(), 2);
    }
    
    #[test]
    fn test_backward_compatibility() {
        // Create a node with single-vector embeddings
        let single_vec_node = MemoryNode::new(vec![1.0, 0.0, 0.0]);
        
        // Create a multi-vector embedding
        let vectors = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
        ];
        
        let multi_vec = MultiVectorEmbedding::new(vectors).unwrap();
        
        // Create a node with multi-vector embeddings
        let multi_vec_node = MemoryNode::with_multi_vector(multi_vec);
        
        // Test the legacy embeddings method
        let single_vec_legacy = single_vec_node.embeddings_legacy();
        let multi_vec_legacy = multi_vec_node.embeddings_legacy();
        
        // For single-vector, should be the same as the original
        assert_eq!(single_vec_legacy, vec![1.0, 0.0, 0.0]);
        
        // For multi-vector, should be a flattened version
        assert_eq!(multi_vec_legacy, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]);
    }
}