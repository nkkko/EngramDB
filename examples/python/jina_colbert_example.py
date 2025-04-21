#!/usr/bin/env python
"""
Jina ColBERT Multi-Vector Embeddings Example for EngramDB

This example demonstrates how to use the Jina ColBERT v2 model for multi-vector embeddings
in EngramDB to perform more precise semantic search through late interaction matching.
"""

import engramdb
import time

def main():
    print("=== EngramDB Jina ColBERT Multi-Vector Example ===")
    
    # Initialize the embedding service with the Jina ColBERT v2 model
    print("Initializing embedding service with Jina ColBERT v2 model...")
    service = engramdb.EmbeddingService.with_multi_vector_model(
        engramdb.EmbeddingModel.JinaColBERTv2
    )
    
    print(f"Initialized ColBERT service:")
    print(f"  Single-vector dimensions: {service.dimensions()}")
    print(f"  Multi-vector dimensions: {service.multi_vector_dimensions()}")
    print(f"  Has multi-vector capability: {service.has_multi_vector()}")
    
    # Example documents for the knowledge base
    documents = [
        "EngramDB is a specialized database for agent memory systems.",
        "It supports both single-vector and multi-vector embeddings.",
        "Multi-vector embeddings allow for more precise similarity matching.",
        "The late interaction approach compares individual token embeddings.",
        "ColBERT and ColPali are examples of multi-vector embedding models.",
    ]
    
    # Create nodes with multi-vector embeddings
    print("\nCreating memory nodes with ColBERT multi-vector embeddings...")
    multi_vector_nodes = []
    
    for i, doc in enumerate(documents):
        start_time = time.time()
        
        # Generate multi-vector embeddings
        multi_vec = service.generate_multi_vector_for_document(doc, None)
        
        # Create a memory node with multi-vector embeddings
        node = engramdb.MemoryNode.with_multi_vector(multi_vec)
        node.set_attribute("content", engramdb.AttributeValue.String(doc))
        multi_vector_nodes.append(node)
        
        gen_time = time.time() - start_time
        
        print(f"  Document {i+1}: Created multi-vector node in {gen_time:.2f}s")
        print(f"    - {multi_vec.num_vectors()} token vectors of {multi_vec.dimensions()} dimensions each")
        print(f"    - Content: \"{doc}\"")
    
    # Create a database with multi-vector index
    print("\nCreating database with multi-vector index...")
    
    # Configure database for multi-vector search
    db_config = engramdb.DatabaseConfig.new_in_memory()
    db_config.set_vector_algorithm(engramdb.VectorAlgorithm.MultiVector)
    
    # Set multi-vector config with late interaction similarity
    mv_config = engramdb.MultiVectorIndexConfig()
    mv_config.set_similarity_method(engramdb.MultiVectorSimilarityMethod.LateInteraction)
    db_config.set_multi_vector_config(mv_config)
    
    db = engramdb.Database.with_config(db_config)
    
    # Add nodes to database
    for node in multi_vector_nodes:
        db.add_memory(node)
    
    print(f"Added {len(multi_vector_nodes)} nodes to database")
    
    # Perform queries
    queries = [
        "What are multi-vector embeddings?",
        "Tell me about the late interaction approach",
        "What is EngramDB used for?",
    ]
    
    print("\nPerforming ColBERT multi-vector searches:")
    
    for query in queries:
        print(f"\nQuery: \"{query}\"")
        
        # Generate multi-vector query embeddings
        start_time = time.time()
        query_multi_vec = service.generate_multi_vector_for_query(query)
        gen_time = time.time() - start_time
        
        print(f"  Generated query embedding with {query_multi_vec.num_vectors()} token vectors in {gen_time:.2f}s")
        
        # For search compatibility, use the VectorSearchIndex directly
        # We need to use one of the vectors from our multi-vector
        start_time = time.time()
        index = db.get_vector_index()
        
        # For direct multi-vector search, we need to access the underlying multi-vector index
        # This example temporarily uses single vector for compatibility with the public API
        # In a production application, you would access the multi-vector index directly
        results = index.search(query_multi_vec.vectors()[0], 3, 0.0)
        search_time = time.time() - start_time
        
        print(f"  Search completed in {search_time:.2f}s, found {len(results)} results:")
        
        for i, (id, score) in enumerate(results):
            # Look up the node in our local collection
            # In a real application, you'd use db.get_memory(id) or similar
            node = next((n for n in multi_vector_nodes if n.id() == id), None)
            if node:
                content = node.get_attribute("content").as_string()
                print(f"    {i+1}. Score: {score:.4f} - \"{content}\"")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()