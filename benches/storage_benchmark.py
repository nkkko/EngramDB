#!/usr/bin/env python3
"""
Benchmark for storage operations in EngramDB

This benchmark measures:
1. Time to store nodes individually
2. Time to store nodes in batches
3. Time to perform operations with varying database sizes
4. Comparison of in-memory vs. file-based storage
"""
import os
import sys
import time
import argparse
import tempfile
import shutil
from typing import Dict, List, Any, Optional

# Import benchmark utilities
from benchmark_utils import (
    time_function, create_database, create_test_nodes,
    format_benchmark_results, save_benchmark_results,
    print_summary, generate_random_vector
)

def parse_args():
    """Parse command line arguments specific to this benchmark"""
    parser = argparse.ArgumentParser(description="EngramDB Storage Benchmark")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of benchmark iterations (default: 3)")
    parser.add_argument("--max-size", type=int, default=10000,
                        help="Maximum database size for scaling tests (default: 10000)")
    parser.add_argument("--batch-sizes", type=str, default="10,100,1000",
                        help="Comma-separated list of batch sizes to test (default: 10,100,1000)")
    parser.add_argument("--vector-dim", type=int, default=384,
                        help="Vector dimensions (default: 384)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for results (default: None)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    return parser.parse_args()

def run_storage_benchmark(args):
    """Run the storage benchmark"""
    # Parse batch sizes
    batch_sizes = [int(size) for size in args.batch_sizes.split(",")]
    
    results = {
        "in_memory_individual_store": [],
        "file_based_individual_store": [],
        "in_memory_individual_load": [],
        "file_based_individual_load": []
    }
    
    # Add batch results
    for batch_size in batch_sizes:
        results[f"in_memory_batch_{batch_size}"] = []
        results[f"file_based_batch_{batch_size}"] = []
    
    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}")
        
        # Test individual operations for both storage types
        for storage_type in ["memory", "file"]:
            prefix = "in_memory" if storage_type == "memory" else "file_based"
            
            # Create temporary directory for file storage
            temp_dir = None
            if storage_type == "file":
                temp_dir = tempfile.mkdtemp(prefix="engramdb_bench_")
                db_path = os.path.join(temp_dir, "test_db")
                db = engramdb_py.Database.file_based(db_path)
            else:
                db = engramdb_py.Database.in_memory()
            
            try:
                # Benchmark individual storage
                nodes = create_test_nodes(100, args.vector_dim)
                for node in nodes:
                    _, time_ms = time_function(db.save, node)
                    results[f"{prefix}_individual_store"].append(time_ms)
                
                # Get all node IDs
                node_ids = db.list_all()
                
                # Benchmark individual loading
                for node_id in node_ids:
                    _, time_ms = time_function(db.load, node_id)
                    results[f"{prefix}_individual_load"].append(time_ms)
                
                # Clear database
                db.clear_all()
                
                # Benchmark batch operations for each batch size
                for batch_size in batch_sizes:
                    nodes = create_test_nodes(batch_size, args.vector_dim)
                    
                    # Time how long it takes to store the entire batch
                    start_time = time.time()
                    for node in nodes:
                        db.save(node)
                    end_time = time.time()
                    
                    batch_time_ms = (end_time - start_time) * 1000.0
                    results[f"{prefix}_batch_{batch_size}"].append(batch_time_ms)
                    
                    # Clear database for next batch
                    db.clear_all()
            
            finally:
                # Clean up
                if temp_dir:
                    shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Format and save results
    formatted_results = format_benchmark_results("Storage Benchmark", results)
    formatted_results["config"] = {
        "max_size": args.max_size,
        "batch_sizes": batch_sizes,
        "vector_dimensions": args.vector_dim,
    }
    
    save_benchmark_results(formatted_results, args.output, args.append)
    print_summary(formatted_results)
    
    return formatted_results

if __name__ == "__main__":
    args = parse_args()
    
    # Import here to avoid circular imports
    import engramdb_py
    
    run_storage_benchmark(args)