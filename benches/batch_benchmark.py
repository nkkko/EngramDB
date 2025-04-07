#!/usr/bin/env python3
"""
Benchmark for batch operations in EngramDB

This benchmark compares:
1. Individual node operations vs. batch operations
2. Performance of batch operations with varying batch sizes
3. Performance implications of different operations (save, load, delete)
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
    print_summary
)

def parse_args():
    """Parse command line arguments specific to this benchmark"""
    parser = argparse.ArgumentParser(description="EngramDB Batch Operations Benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations (default: 5)")
    parser.add_argument("--batch-sizes", type=str, default="10,50,100,500,1000",
                        help="Comma-separated list of batch sizes (default: 10,50,100,500,1000)")
    parser.add_argument("--vector-dim", type=int, default=384,
                        help="Vector dimensions (default: 384)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for results (default: None)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    return parser.parse_args()

def run_batch_benchmark(args):
    """Run the batch operations benchmark"""
    # Parse batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    results = {
        "individual_save": [],
        "individual_load": [],
        "individual_delete": []
    }
    
    # Add batch results
    for batch_size in batch_sizes:
        results[f"batch_save_{batch_size}"] = []
        results[f"batch_load_{batch_size}"] = []
        results[f"batch_delete_{batch_size}"] = []
    
    # Import here to avoid circular imports
    import engramdb_py as engramdb
    
    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}")
        
        # Create in-memory database
        db = engramdb.Database.in_memory()
        
        # Create test nodes for the largest batch size
        max_batch_size = max(batch_sizes)
        nodes = create_test_nodes(max_batch_size, args.vector_dim)
        
        # Test individual operations
        print("Testing individual operations...")
        
        # Individual save
        individual_save_start = time.time()
        node_ids = []
        for node in nodes[:100]:  # Use a smaller subset for individual operations
            node_id = db.save(node)
            node_ids.append(node_id)
        individual_save_end = time.time()
        results["individual_save"].append((individual_save_end - individual_save_start) * 1000.0 / 100)
        
        # Individual load
        individual_load_start = time.time()
        for node_id in node_ids:
            node = db.load(node_id)
        individual_load_end = time.time()
        results["individual_load"].append((individual_load_end - individual_load_start) * 1000.0 / len(node_ids))
        
        # Individual delete
        individual_delete_start = time.time()
        for node_id in node_ids:
            db.delete(node_id)
        individual_delete_end = time.time()
        results["individual_delete"].append((individual_delete_end - individual_delete_start) * 1000.0 / len(node_ids))
        
        # Test batch operations for each batch size
        for batch_size in batch_sizes:
            print(f"Testing batch size {batch_size}...")
            
            # Clear database
            db.clear_all()
            
            # Batch save
            test_nodes = nodes[:batch_size]
            batch_save_start = time.time()
            node_ids = []
            for node in test_nodes:
                node_id = db.save(node)
                node_ids.append(node_id)
            batch_save_end = time.time()
            results[f"batch_save_{batch_size}"].append((batch_save_end - batch_save_start) * 1000.0)
            
            # Batch load
            batch_load_start = time.time()
            loaded_nodes = []
            for node_id in node_ids:
                node = db.load(node_id)
                loaded_nodes.append(node)
            batch_load_end = time.time()
            results[f"batch_load_{batch_size}"].append((batch_load_end - batch_load_start) * 1000.0)
            
            # Batch delete
            batch_delete_start = time.time()
            for node_id in node_ids:
                db.delete(node_id)
            batch_delete_end = time.time()
            results[f"batch_delete_{batch_size}"].append((batch_delete_end - batch_delete_start) * 1000.0)
    
    # Calculate average time per node for batch operations
    for batch_size in batch_sizes:
        batch_save_times = results[f"batch_save_{batch_size}"]
        results[f"batch_save_{batch_size}_per_node"] = [time / batch_size for time in batch_save_times]
        
        batch_load_times = results[f"batch_load_{batch_size}"]
        results[f"batch_load_{batch_size}_per_node"] = [time / batch_size for time in batch_load_times]
        
        batch_delete_times = results[f"batch_delete_{batch_size}"]
        results[f"batch_delete_{batch_size}_per_node"] = [time / batch_size for time in batch_delete_times]
    
    # Format and save results
    formatted_results = format_benchmark_results("Batch Operations Benchmark", results)
    formatted_results["config"] = {
        "batch_sizes": batch_sizes,
        "vector_dimensions": args.vector_dim,
    }
    
    save_benchmark_results(formatted_results, args.output, args.append)
    print_summary(formatted_results)
    
    return formatted_results

if __name__ == "__main__":
    args = parse_args()
    run_batch_benchmark(args)