"""
Utilities for EngramDB benchmarks
"""
import os
import sys
import time
import json
import random
import string
import argparse
import tempfile
import shutil
from typing import Dict, List, Any, Optional, Callable, Tuple
from datetime import datetime
import numpy as np

# Add parent directory to path to find engramdb_py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    import engramdb_py as engramdb
except ImportError:
    print("Error: Could not import engramdb_py")
    print("Make sure you've installed the Python bindings: pip install -e python/")
    sys.exit(1)

# Define benchmark result structure
BenchmarkResult = Dict[str, Any]

def parse_common_args() -> argparse.Namespace:
    """Parse common command line arguments for benchmarks"""
    parser = argparse.ArgumentParser(description="EngramDB Benchmark")
    parser.add_argument("--iterations", type=int, default=5,
                        help="Number of benchmark iterations (default: 5)")
    parser.add_argument("--dataset-size", type=int, default=1000,
                        help="Size of test dataset (default: 1000)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output file path for results (default: None)")
    parser.add_argument("--append", action="store_true",
                        help="Append to output file instead of overwriting")
    return parser.parse_args()

def create_database(storage_type: str = "memory", cleanup_func=None) -> Tuple[engramdb.Database, Optional[str]]:
    """
    Create a database for testing
    
    Args:
        storage_type: "memory" or "file"
        cleanup_func: Function to call for cleanup if needed
        
    Returns:
        Tuple of (database, path)
    """
    if storage_type == "memory":
        return engramdb.Database.in_memory(), None
    else:
        temp_dir = tempfile.mkdtemp(prefix="engramdb_bench_")
        db_path = os.path.join(temp_dir, "test_db")
        
        # Register cleanup if provided
        if cleanup_func:
            cleanup_func(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
            
        return engramdb.Database.file_based(db_path), db_path

def generate_random_text(min_words: int = 5, max_words: int = 50) -> str:
    """
    Generate random text for benchmark testing
    
    Args:
        min_words: Minimum number of words
        max_words: Maximum number of words
        
    Returns:
        Random text string
    """
    word_count = random.randint(min_words, max_words)
    words = []
    
    for _ in range(word_count):
        word_length = random.randint(2, 10)
        word = ''.join(random.choice(string.ascii_lowercase) for _ in range(word_length))
        words.append(word)
        
    return ' '.join(words)

def generate_random_vector(dimensions: int = 384, normalized: bool = True) -> List[float]:
    """
    Generate a random vector for benchmark testing
    
    Args:
        dimensions: Number of dimensions for the vector
        normalized: Whether to normalize the vector
        
    Returns:
        Random vector as a list of floats
    """
    vector = np.random.normal(0, 1, dimensions).astype(np.float32)
    
    if normalized:
        # Normalize to unit length
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
            
    return vector.tolist()

def time_function(func: Callable, *args, **kwargs) -> Tuple[Any, float]:
    """
    Time the execution of a function
    
    Args:
        func: Function to time
        *args, **kwargs: Arguments to pass to function
        
    Returns:
        Tuple of (result, execution_time_in_ms)
    """
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    
    execution_time_ms = (end_time - start_time) * 1000.0
    return result, execution_time_ms

def create_test_nodes(count: int, vector_dim: int = 384) -> List[engramdb.MemoryNode]:
    """
    Create test memory nodes for benchmarking
    
    Args:
        count: Number of nodes to create
        vector_dim: Dimensions of node vectors
        
    Returns:
        List of MemoryNode objects
    """
    nodes = []
    for i in range(count):
        text = generate_random_text(10, 100)
        vector = generate_random_vector(vector_dim)
        
        node = engramdb.MemoryNode(vector)
        node.set_attribute("title", f"Test Node {i}")
        node.set_attribute("content", text)
        node.set_attribute("category", random.choice(["note", "task", "contact", "document"]))
        node.set_attribute("timestamp", datetime.now().isoformat())
        
        nodes.append(node)
        
    return nodes

def format_benchmark_results(benchmark_name: str, results: Dict[str, List[float]]) -> BenchmarkResult:
    """
    Format benchmark results for reporting
    
    Args:
        benchmark_name: Name of the benchmark
        results: Dictionary mapping test names to lists of timing results
        
    Returns:
        Formatted benchmark result
    """
    formatted_results = {
        "benchmark_name": benchmark_name,
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "python_version": sys.version,
            "platform": sys.platform
        },
        "tests": {}
    }
    
    for test_name, timings in results.items():
        if not timings:
            continue
            
        formatted_results["tests"][test_name] = {
            "mean_ms": np.mean(timings),
            "median_ms": np.median(timings),
            "min_ms": np.min(timings),
            "max_ms": np.max(timings),
            "std_ms": np.std(timings),
            "iterations": len(timings),
            "raw_timings_ms": timings
        }
        
    return formatted_results

def save_benchmark_results(results: BenchmarkResult, output_path: Optional[str] = None, append: bool = False) -> None:
    """
    Save benchmark results to a file or print to stdout
    
    Args:
        results: Benchmark results dictionary
        output_path: Path to output file, or None for stdout
        append: Whether to append to existing file
    """
    if output_path:
        mode = 'a' if append else 'w'
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        
        with open(output_path, mode) as f:
            f.write(json.dumps(results, indent=2))
            f.write('\n')
            print(f"Results saved to {output_path}")
    else:
        # Print to stdout
        print(json.dumps(results, indent=2))
        
def print_summary(results: BenchmarkResult) -> None:
    """
    Print a human-readable summary of benchmark results
    
    Args:
        results: Benchmark results dictionary
    """
    print(f"\n=== {results['benchmark_name']} ===")
    print(f"Timestamp: {results['timestamp']}")
    print("\nTest Results:")
    
    for test_name, stats in results["tests"].items():
        print(f"\n{test_name}:")
        print(f"  Mean:   {stats['mean_ms']:.2f} ms")
        print(f"  Median: {stats['median_ms']:.2f} ms")
        print(f"  Min:    {stats['min_ms']:.2f} ms")
        print(f"  Max:    {stats['max_ms']:.2f} ms")
        print(f"  Std:    {stats['std_ms']:.2f} ms")