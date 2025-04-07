#!/usr/bin/env python3
"""
Benchmark for vector search operations in EngramDB

This benchmark measures:
1. Time to perform similarity searches with varying database sizes
2. Time to search with different thresholds
3. Impact of vector dimensionality on search performance
"""
import os
import sys
import time
import argparse
import numpy as np
from typing import Dict, List, Any, Optional

# Import benchmark utilities
from benchmark_utils import (
    time_function, create_database, create_test_nodes,
    format_benchmark_results, save_benchmark_results,
    print_summary, generate_random_vector
)

def parse_args():
    """Parse command line arguments specific to this benchmark"""
    parser = argparse.ArgumentParser(description="EngramDB Search Benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations (default: 5)")
    parser.add_argument("--dataset-sizes", type=str, default="100,1000,10000",
                        help="Comma-separated list of dataset sizes (default: 100,1000,10000)")
    parser.add_argument("--thresholds", type=str, default="0.5,0.7,0.9",
                        help="Comma-separated list of search thresholds (default: 0.5,0.7,0.9)")
    parser.add_argument("--vector-dims", type=str, default="64,128,384,768",
                        help="Comma-separated list of vector dimensions to test (default: 64,128,384,768)")
    parser.add_argument("--limit", type=int, default=10,
                        help="Number of results to return from search (default: 10)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for results (default: None)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    return parser.parse_args()

def run_search_benchmark(args):
    """Run the search benchmark"""
    # Parse argument lists
    dataset_sizes = [int(size) for size in args.dataset_sizes.split(",")]
    thresholds = [float(t) for t in args.thresholds.split(",")]
    vector_dims = [int(dim) for dim in args.vector_dims.split(",")]
    
    results = {}
    
    # Initialize result containers
    for size in dataset_sizes:
        results[f"search_size_{size}"] = []
    
    for threshold in thresholds:
        results[f"search_threshold_{threshold}"] = []
    
    for dim in vector_dims:
        results[f"search_dim_{dim}"] = []
    
    # Import here to avoid circular imports
    import engramdb_py as engramdb
    
    # Benchmark search with varying database sizes
    print("\nBenchmarking search with varying database sizes...")
    for size in dataset_sizes:
        for i in range(args.iterations):
            print(f"Testing size {size}, iteration {i+1}/{args.iterations}")
            
            # Create in-memory database
            db = engramdb.Database.in_memory()
            
            # Create and store nodes
            nodes = create_test_nodes(size, vector_dim=384)
            node_ids = []
            for node in nodes:
                node_id = db.save(node)
                node_ids.append(node_id)
            
            # Generate query vector
            query_vector = generate_random_vector(384)
            
            # Time the search operation
            _, time_ms = time_function(
                db.search_similar, 
                query_vector, 
                limit=args.limit, 
                threshold=0.5
            )
            
            results[f"search_size_{size}"].append(time_ms)
    
    # Benchmark search with fixed size but varying thresholds
    print("\nBenchmarking search with varying thresholds...")
    fixed_size = 1000
    db = engramdb.Database.in_memory()
    
    # Create and store nodes once for threshold tests
    nodes = create_test_nodes(fixed_size, vector_dim=384)
    for node in nodes:
        db.save(node)
    
    for threshold in thresholds:
        for i in range(args.iterations):
            print(f"Testing threshold {threshold}, iteration {i+1}/{args.iterations}")
            
            # Generate query vector
            query_vector = generate_random_vector(384)
            
            # Time the search operation
            _, time_ms = time_function(
                db.search_similar, 
                query_vector, 
                limit=args.limit, 
                threshold=threshold
            )
            
            results[f"search_threshold_{threshold}"].append(time_ms)
    
    # Benchmark search with varying vector dimensions
    print("\nBenchmarking search with varying vector dimensions...")
    for dim in vector_dims:
        for i in range(args.iterations):
            print(f"Testing dimension {dim}, iteration {i+1}/{args.iterations}")
            
            # Create new database for each dimension test
            db = engramdb.Database.in_memory()
            
            # Create and store nodes with specific vector dimension
            nodes = create_test_nodes(500, vector_dim=dim)
            for node in nodes:
                db.save(node)
            
            # Generate query vector with matching dimensions
            query_vector = generate_random_vector(dim)
            
            # Time the search operation
            _, time_ms = time_function(
                db.search_similar, 
                query_vector, 
                limit=args.limit, 
                threshold=0.5
            )
            
            results[f"search_dim_{dim}"].append(time_ms)
    
    # Format and save results
    formatted_results = format_benchmark_results("Search Benchmark", results)
    formatted_results["config"] = {
        "dataset_sizes": dataset_sizes,
        "thresholds": thresholds,
        "vector_dims": vector_dims,
        "limit": args.limit
    }
    
    save_benchmark_results(formatted_results, args.output, args.append)
    print_summary(formatted_results)
    
    return formatted_results

if __name__ == "__main__":
    args = parse_args()
    run_search_benchmark(args)