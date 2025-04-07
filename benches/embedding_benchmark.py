#!/usr/bin/env python3
"""
Benchmark for embedding generation and storage in EngramDB

This benchmark measures:
1. Time to generate embeddings from text
2. Time to create EngramDB nodes with embeddings
3. Time to store these nodes in the database
4. Time to retrieve and load nodes
"""
import os
import sys
import argparse
import random
from typing import Dict, List, Any, Optional

# Import benchmark utilities
from benchmark_utils import (
    time_function, create_database, create_test_nodes,
    format_benchmark_results, save_benchmark_results,
    print_summary, parse_common_args, generate_random_text
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../web')))
try:
    from embedding_utils import generate_embeddings, mock_embeddings
except ImportError:
    print("Warning: Could not import embedding_utils from web directory.")
    print("Will use mock embeddings only.")
    generate_embeddings = None
    
    # Define mock_embeddings if not imported
    def mock_embeddings(dimensions=384):
        import numpy as np
        embedding = np.random.normal(0, 1, dimensions)
        embedding = embedding / np.linalg.norm(embedding)
        return embedding

def parse_args():
    """Parse command line arguments specific to this benchmark"""
    parser = argparse.ArgumentParser(description="EngramDB Embedding Benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations (default: 5)")
    parser.add_argument("--dataset-size", type=int, default=100,
                        help="Size of test dataset (default: 100)")
    parser.add_argument("--vector-dim", type=int, default=384,
                        help="Vector dimensions for mock embeddings (default: 384)")
    parser.add_argument("--use-real-embeddings", action="store_true",
                        help="Use real embedding model instead of mock embeddings")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for results (default: None)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    return parser.parse_args()

def run_embedding_benchmark(args):
    """Run the embedding benchmark"""
    results = {
        "embedding_generation": [],
        "node_creation": [],
        "node_storage": [],
        "node_retrieval": []
    }
    
    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}")
        
        # Create database
        db, db_path = create_database("memory")
        
        # Generate test texts
        texts = [generate_random_text(20, 200) for _ in range(args.dataset_size)]
        
        # Benchmark embedding generation
        embeddings = []
        for text in texts:
            if args.use_real_embeddings and generate_embeddings:
                embedding_func = lambda t: generate_embeddings(t, "Represent the document for retrieval")
            else:
                embedding_func = lambda _: mock_embeddings(args.vector_dim)
                
            embedding, time_ms = time_function(embedding_func, text)
            embeddings.append(embedding)
            results["embedding_generation"].append(time_ms)
        
        # Benchmark node creation
        nodes = []
        for j, (text, embedding) in enumerate(zip(texts, embeddings)):
            node_func = lambda: create_node_with_embedding(text, embedding, j)
            node, time_ms = time_function(node_func)
            nodes.append(node)
            results["node_creation"].append(time_ms)
        
        # Benchmark node storage
        node_ids = []
        for node in nodes:
            storage_func = lambda n: db.save(n)
            node_id, time_ms = time_function(storage_func, node)
            node_ids.append(node_id)
            results["node_storage"].append(time_ms)
        
        # Benchmark node retrieval
        for node_id in node_ids:
            retrieval_func = lambda id: db.load(id)
            _, time_ms = time_function(retrieval_func, node_id)
            results["node_retrieval"].append(time_ms)
    
    # Format and save results
    benchmark_name = "Embedding Benchmark"
    if args.use_real_embeddings and generate_embeddings:
        benchmark_name += " (Real Embeddings)"
    else:
        benchmark_name += " (Mock Embeddings)"
        
    formatted_results = format_benchmark_results(benchmark_name, results)
    formatted_results["config"] = {
        "dataset_size": args.dataset_size,
        "vector_dimensions": args.vector_dim,
        "using_real_embeddings": args.use_real_embeddings and generate_embeddings is not None
    }
    
    save_benchmark_results(formatted_results, args.output, args.append)
    print_summary(formatted_results)
    
    return formatted_results

def create_node_with_embedding(text, embedding, index):
    """Create a memory node with the provided embedding"""
    import engramdb_py as engramdb
    
    node = engramdb.MemoryNode(embedding)
    node.set_attribute("title", f"Embedding Test {index}")
    node.set_attribute("content", text)
    node.set_attribute("category", random.choice(["note", "task", "contact", "document"]))
    
    return node

if __name__ == "__main__":
    args = parse_args()
    run_embedding_benchmark(args)