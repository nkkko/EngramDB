#!/usr/bin/env python3
"""
Benchmark comparing in-memory vs. file-based storage in EngramDB

This benchmark compares:
1. Basic operation speed (save, load, delete) between storage types
2. Performance scaling with increasing database size
3. Trade-offs between memory usage and speed
"""
import os
import sys
import time
import tempfile
import shutil
import argparse
import psutil
import numpy as np
from typing import Dict, List, Any, Optional, Tuple

# Import benchmark utilities
from benchmark_utils import (
    time_function, create_database, create_test_nodes,
    format_benchmark_results, save_benchmark_results,
    print_summary
)

def parse_args():
    """Parse command line arguments specific to this benchmark"""
    parser = argparse.ArgumentParser(description="EngramDB Memory vs. File Storage Benchmark")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of benchmark iterations (default: 3)")
    parser.add_argument("--dataset-sizes", type=str, default="100,1000,10000",
                        help="Comma-separated list of dataset sizes (default: 100,1000,10000)")
    parser.add_argument("--vector-dim", type=int, default=384,
                        help="Vector dimensions (default: 384)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for results (default: None)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    return parser.parse_args()

def get_memory_usage():
    """Get current memory usage in MB"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    return memory_info.rss / (1024 * 1024)  # Convert to MB

def run_memory_vs_file_benchmark(args):
    """Run the memory vs. file storage benchmark"""
    # Parse dataset sizes
    dataset_sizes = [int(size) for size in args.dataset_sizes.split(",")]
    
    results = {}
    
    # Initialize result containers
    for size in dataset_sizes:
        for storage_type in ["memory", "file"]:
            prefix = "memory" if storage_type == "memory" else "file"
            results[f"{prefix}_save_{size}"] = []
            results[f"{prefix}_load_{size}"] = []
            results[f"{prefix}_search_{size}"] = []
            results[f"{prefix}_mem_usage_{size}"] = []
            results[f"{prefix}_delete_{size}"] = []
    
    # Import here to avoid circular imports
    import engramdb_py as engramdb
    
    for i in range(args.iterations):
        print(f"Running iteration {i+1}/{args.iterations}")
        
        for size in dataset_sizes:
            print(f"Testing dataset size {size}...")
            
            # Test each storage type
            for storage_type in ["memory", "file"]:
                prefix = "memory" if storage_type == "memory" else "file"
                print(f"  Testing {storage_type} storage...")
                
                # Create database
                temp_dir = None
                if storage_type == "file":
                    temp_dir = tempfile.mkdtemp(prefix="engramdb_bench_")
                    db_path = os.path.join(temp_dir, "test_db")
                    db = engramdb.Database.file_based(db_path)
                else:
                    db = engramdb.Database.in_memory()
                
                try:
                    # Create test nodes
                    nodes = create_test_nodes(size, args.vector_dim)
                    
                    # Measure save performance
                    save_start = time.time()
                    node_ids = []
                    for node in nodes:
                        node_id = db.save(node)
                        node_ids.append(node_id)
                    save_end = time.time()
                    results[f"{prefix}_save_{size}"].append((save_end - save_start) * 1000.0)
                    
                    # Measure memory usage after saving
                    mem_usage = get_memory_usage()
                    results[f"{prefix}_mem_usage_{size}"].append(mem_usage)
                    
                    # Measure load performance
                    load_start = time.time()
                    for node_id in node_ids[:100]:  # Load a subset to keep benchmark reasonable
                        node = db.load(node_id)
                    load_end = time.time()
                    results[f"{prefix}_load_{size}"].append((load_end - load_start) * 1000.0 / 100)
                    
                    # Measure search performance
                    search_vector = np.random.normal(0, 1, args.vector_dim).astype(np.float32)
                    search_vector = search_vector / np.linalg.norm(search_vector)
                    
                    search_start = time.time()
                    search_results = db.search_similar(search_vector.tolist(), limit=10, threshold=0.5)
                    search_end = time.time()
                    results[f"{prefix}_search_{size}"].append((search_end - search_start) * 1000.0)
                    
                    # Measure delete performance
                    delete_start = time.time()
                    for node_id in node_ids[:100]:  # Delete a subset to keep benchmark reasonable
                        db.delete(node_id)
                    delete_end = time.time()
                    results[f"{prefix}_delete_{size}"].append((delete_end - delete_start) * 1000.0 / 100)
                
                finally:
                    # Clean up
                    if temp_dir:
                        shutil.rmtree(temp_dir, ignore_errors=True)
    
    # Format and save results
    formatted_results = format_benchmark_results("Memory vs. File Storage Benchmark", results)
    formatted_results["config"] = {
        "dataset_sizes": dataset_sizes,
        "vector_dimensions": args.vector_dim,
    }
    
    save_benchmark_results(formatted_results, args.output, args.append)
    print_summary(formatted_results)
    
    return formatted_results

if __name__ == "__main__":
    args = parse_args()
    
    # Check for psutil
    try:
        import psutil
    except ImportError:
        print("Warning: psutil module not found. Memory usage tracking will be disabled.")
        print("Install it with: pip install psutil")
        
        # Redefine get_memory_usage to return a dummy value
        def get_memory_usage():
            return 0.0
    
    run_memory_vs_file_benchmark(args)