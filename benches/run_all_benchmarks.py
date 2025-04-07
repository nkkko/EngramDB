#!/usr/bin/env python3
"""
Script to run all EngramDB benchmarks and aggregate results
"""
import os
import sys
import time
import json
import argparse
import datetime
from typing import Dict, List, Any

def parse_args():
    """Parse command line arguments for running all benchmarks"""
    parser = argparse.ArgumentParser(description="Run all EngramDB benchmarks")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                        help="Directory to store benchmark results (default: benchmark_results)")
    parser.add_argument("--iterations", type=int, default=3,
                        help="Number of iterations for each benchmark (default: 3)")
    parser.add_argument("--skip", type=str, default="",
                        help="Comma-separated list of benchmarks to skip")
    return parser.parse_args()

def run_all_benchmarks(args):
    """Run all benchmarks and save results"""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate timestamp for this run
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Define benchmarks to run
    benchmarks = [
        {
            "name": "embedding",
            "script": "embedding_benchmark.py",
            "args": f"--iterations {args.iterations} --dataset-size 100"
        },
        {
            "name": "storage",
            "script": "storage_benchmark.py",
            "args": f"--iterations {args.iterations} --max-size 1000"
        },
        {
            "name": "search",
            "script": "search_benchmark.py",
            "args": f"--iterations {args.iterations} --dataset-sizes 100,1000"
        },
        {
            "name": "batch",
            "script": "batch_benchmark.py",
            "args": f"--iterations {args.iterations} --batch-sizes 10,100,1000"
        },
        {
            "name": "memory_vs_file",
            "script": "memory_vs_file_benchmark.py",
            "args": f"--iterations {args.iterations} --dataset-sizes 100,1000"
        }
    ]
    
    # Check for benchmarks to skip
    skip_benchmarks = [b.strip() for b in args.skip.split(",")] if args.skip else []
    
    # Store all results
    all_results = {}
    
    # Run each benchmark
    for benchmark in benchmarks:
        if benchmark["name"] in skip_benchmarks:
            print(f"Skipping {benchmark['name']} benchmark")
            continue
        
        print(f"\n{'=' * 80}")
        print(f"Running {benchmark['name']} benchmark")
        print(f"{'=' * 80}")
        
        output_file = os.path.join(args.output_dir, f"{benchmark['name']}_{timestamp}.json")
        
        # Construct command
        cmd = f"python {benchmark['script']} {benchmark['args']} --output {output_file}"
        print(f"Command: {cmd}")
        
        # Run benchmark
        start_time = time.time()
        exit_code = os.system(cmd)
        end_time = time.time()
        
        if exit_code != 0:
            print(f"Error running {benchmark['name']} benchmark, exit code: {exit_code}")
            continue
        
        print(f"Completed in {end_time - start_time:.2f} seconds")
        
        # Load and store results
        try:
            with open(output_file, 'r') as f:
                results = json.load(f)
                all_results[benchmark['name']] = results
        except Exception as e:
            print(f"Error loading results from {output_file}: {e}")
    
    # Save aggregated results
    aggregated_file = os.path.join(args.output_dir, f"all_benchmarks_{timestamp}.json")
    with open(aggregated_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"All benchmarks completed, results saved to {args.output_dir}")
    print(f"Aggregated results: {aggregated_file}")
    print(f"{'=' * 80}")

if __name__ == "__main__":
    args = parse_args()
    run_all_benchmarks(args)