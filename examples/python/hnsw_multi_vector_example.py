#!/usr/bin/env python
"""
HNSW Multi-Vector Embeddings Example for EngramDB

This example demonstrates how to use the HNSW-optimized multi-vector embeddings
in EngramDB to perform more precise semantic search with better performance.
"""

import engramdb
import time
import random

def main():
    print("=== EngramDB HNSW Multi-Vector Embeddings Example (Python) ===\n")
    
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
        "HNSW (Hierarchical Navigable Small World) is a graph-based algorithm for approximate nearest neighbor search.",
        "HNSW-based multi-vector search combines token-level precision with graph-based efficiency.",
        "The two-phase approach uses a representative vector for efficient candidate retrieval.",
        "Deduplication of shared vectors can significantly reduce memory usage in token-based embeddings.",
        "Different pooling strategies can be used to create representative vectors for the graph.",
    ]
    
    # Create a linear multi-vector database (for comparison)
    print("\nCreating linear multi-vector database...")
    db_linear = engramdb.Database.in_memory_with_multi_vector()
    
    # Create an HNSW-based multi-vector database
    print("Creating HNSW multi-vector database...")
    db_hnsw = engramdb.Database.in_memory_with_hnsw_multi_vector()
    
    # Add documents to both databases
    print("\nAdding documents with multi-vector embeddings...")
    
    for i, doc in enumerate(documents):
        # Create a multi-vector node
        node = engramdb.MemoryNode.from_text_multi_vector(doc, service, None)
        
        # Add to both databases
        db_linear.save(node)
        db_hnsw.save(node)
        
        print(f"- Added document {i+1} with multi-vector embeddings")
    
    # Add more documents with random content to test scaling
    num_extra_docs = 290  # Add more to test performance at scale
    print(f"\nAdding {num_extra_docs} additional documents for performance testing...")
    
    # Generate some random documents
    word_list = [
        "vector", "embedding", "similarity", "search", "database", "memory", 
        "agent", "token", "graph", "neural", "network", "semantic", "query", 
        "retrieval", "precision", "recall", "efficient", "scalable", "knowledge",
        "information", "document", "text", "content", "context", "relevant",
        "index", "storage", "algorithm", "data", "model", "latent", "space"
    ]
    
    for i in range(num_extra_docs):
        # Generate a random document
        words = random.sample(word_list, k=random.randint(5, 15))
        doc = " ".join(words) + "."
        
        # Create a multi-vector node
        node = engramdb.MemoryNode.from_text_multi_vector(doc, service, None)
        
        # Add to both databases
        db_linear.save(node)
        db_hnsw.save(node)
        
        if (i + 1) % 50 == 0:
            print(f"  - Added {i + 1} additional documents...")
    
    # Perform search with both implementations and compare
    print("\nPerforming search tests:")
    query = "Tell me about efficient vector search algorithms"
    
    # Generate multi-vector query embeddings
    query_multi_vec = service.generate_multi_vector_for_query(query)
    print(f"Query: \"{query}\"")
    print(f"Query has {query_multi_vec.num_vectors()} vectors of {query_multi_vec.dimensions()} dimensions")
    
    # Search with linear multi-vector index
    print("\nPerforming search with linear multi-vector index...")
    start_time = time.time()
    results_linear = db_linear.search_by_multi_vector(query_multi_vec, 5, 0.0)
    linear_search_time = time.time() - start_time
    print(f"Search completed in {linear_search_time:.4f} seconds")
    
    # Search with HNSW multi-vector index
    print("\nPerforming search with HNSW multi-vector index...")
    start_time = time.time()
    results_hnsw = db_hnsw.search_by_multi_vector(query_multi_vec, 5, 0.0)
    hnsw_search_time = time.time() - start_time
    print(f"Search completed in {hnsw_search_time:.4f} seconds")
    
    # Calculate and print speedup
    speedup = linear_search_time / hnsw_search_time if hnsw_search_time > 0 else float('inf')
    print(f"\nHNSW multi-vector index is {speedup:.2f}x faster than linear multi-vector index")
    
    # Print top results from HNSW search
    print("\nTop 5 results from HNSW multi-vector search:")
    for i, (node, score) in enumerate(results_hnsw):
        content = node.get_attribute("content").as_string()
        print(f"{i+1}. Score: {score:.4f} - \"{content}\"")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
