# EngramDB Benchmarks

This directory contains benchmark tests for EngramDB, measuring performance across various operations.

## Available Benchmarks

1. **embedding_benchmark.py** - Tests the performance of embedding generation and storage
2. **storage_benchmark.py** - Measures database storage performance with varying node counts
3. **search_benchmark.py** - Tests vector similarity search performance
4. **batch_benchmark.py** - Compares batch operations vs individual operations
5. **memory_vs_file_benchmark.py** - Compares in-memory vs. file-based storage

## Running Benchmarks

You can run individual benchmarks:

```bash
python embedding_benchmark.py
```

Or run all benchmarks:

```bash
python run_all_benchmarks.py
```

## Benchmark Parameters

Most benchmarks accept the following command-line arguments:

- `--iterations`: Number of iterations to run (default: 5)
- `--dataset-size`: Size of the test dataset (default: 1000)
- `--output`: Output file for results (default: prints to stdout)

See individual benchmark files for specific options.