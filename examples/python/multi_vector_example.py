#!/usr/bin/env python
"""
Multi-Vector Embeddings Example for EngramDB

This example demonstrates how to use multi-vector embeddings (ColBERT/ColPali-style)
in EngramDB to perform more precise semantic search through late interaction matching.
"""

import engramdb
import time

def main():
    print("=== EngramDB Multi-Vector Embeddings Example (Python) ===")
    
    # Create an embedding service with multi-vector capability
    # For this example, we'll use the mock providers
    service = engramdb.EmbeddingService.new_mock_multi_vector(96, 20)
    
    print(f"Created embedding service with multi-vector capability")
    print(f"Single-vector dimensions: {service.dimensions()}")
    print(f"Multi-vector dimensions: {service.multi_vector_dimensions()}")
    print(f"Has multi-vector capability: {service.has_multi_vector()}")
    
    # Create example documents
    documents = [
        "EngramDB is a specialized database for agent memory systems.",
        "It supports both single-vector and multi-vector embeddings.",
        "Multi-vector embeddings allow for more precise similarity matching.",
        "The late interaction approach compares individual token embeddings.",
        "ColBERT and ColPali are examples of multi-vector embedding models.",
    ]
    
    # Create both single-vector and multi-vector nodes
    single_vector_nodes = []
    multi_vector_nodes = []
    
    print("\nCreating memory nodes with both embedding types:")
    for i, doc in enumerate(documents):
        # Create a single-vector node
        node_sv = engramdb.MemoryNode.from_text(doc, service, None)
        single_vector_nodes.append(node_sv)
        
        # Create a multi-vector node
        node_mv = engramdb.MemoryNode.from_text_multi_vector(doc, service, None)
        multi_vector_nodes.append(node_mv)
        
        print(f"- Document {i+1}: Created single and multi-vector nodes")
        
        sv_dims = len(node_sv.embeddings_legacy()) if node_sv.embeddings() is not None else 0
        mv_dims = (f"{node_mv.multi_vector_embeddings().num_vectors()} vectors of "
                  f"{node_mv.multi_vector_embeddings().dimensions()} dimensions each")
        
        print(f"  └ Single vector dimensions: {sv_dims}")
        print(f"  └ Multi vector: {mv_dims}")
    
    # Create a database with multi-vector support
    print("\nCreating database with multi-vector index")
    
    # Create a database configured for multi-vector
    db_config = engramdb.DatabaseConfig.new_in_memory()
    db_config.set_vector_algorithm(engramdb.VectorAlgorithm.MultiVector)
    
    # Set multi-vector config with late interaction similarity
    mv_config = engramdb.MultiVectorIndexConfig()
    mv_config.set_similarity_method(engramdb.MultiVectorSimilarityMethod.LateInteraction)
    db_config.set_multi_vector_config(mv_config)
    
    db = engramdb.Database.with_config(db_config)
    
    # Add multi-vector nodes to the database
    print("Adding multi-vector nodes to the database...")
    for node in multi_vector_nodes:
        db.add(node)
    
    # Perform a multi-vector search
    query = "Tell me about vector embeddings"
    print(f"\nSearching for: \"{query}\"")
    
    # Generate multi-vector query embeddings
    query_multi_vec = service.generate_multi_vector_for_query(query)
    print(f"Query has {query_multi_vec.num_vectors()} vectors of {query_multi_vec.dimensions()} dimensions")
    
    # Search with the specialized multi-vector index
    print("\nPerforming search...")
    start_time = time.time()
    
    # We need to use the underlying vector from multi-vector for the API
    # This is a compatibility layer - internally it will use the multi-vector similarity
    query_vec = query_multi_vec.vectors()[0]
    results = db.search_by_vector(query_vec, 3, 0.0)
    
    search_time = time.time() - start_time
    print(f"Search completed in {search_time:.4f} seconds")
    
    print("\nResults:")
    for i, (node, score) in enumerate(results):
        content = node.get_attribute("content").as_string()
        print(f"{i+1}. Score: {score:.4f} - \"{content}\"")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()