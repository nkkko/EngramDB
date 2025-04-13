# EngramDB

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/Rust-1.70%2B-orange.svg)](https://www.rust-lang.org/)
[![Python](https://img.shields.io/badge/Python-3.7%2B-blue.svg)](https://www.python.org/)
[![Documentation](https://img.shields.io/badge/Docs-mintlify-blue)](https://engramdb.mintlify.app/)
[![Version](https://img.shields.io/badge/Version-0.1.0-green)](https://github.com/nkkko/engramdb)

EngramDB is a specialized database designed for agent memory management. It enables efficient storage, retrieval, and querying of agent memories through a unified memory representation model.

📚 **Documentation**: [https://engramdb.mintlify.app/](https://engramdb.mintlify.app/)  
📊 **Pitch Deck**: [EngramDB Pitch Deck (PDF)](deck/engramdb_pitch_deck.pdf)

![EngramDB Architecture Diagram](assets/engram_arch_2.svg)

## Core Features

- **Unified Memory Representation**: Combines graph, vector, and temporal properties in a single primitive
- **Vector Similarity Search**: Find semantically similar memories with efficient similarity metrics
  - **HNSW Vector Index**: Hierarchical Navigable Small World algorithm for super-fast search
  - **Multiple Embedding Models**: Support for E5, GTE, Jina, and custom embedding models
- **Flexible Storage Options**: Choose between in-memory (for testing) or file-based (for persistence) storage
- **Advanced Query System**: Rich query capabilities with vector similarity, attribute filters, and temporal constraints
- **Memory Evolution**: Track memory changes over time with temporal versioning
- **First-class Python Bindings**: Seamless integration with ML/AI applications
- **Web Interface**: Browser-based visualization and interaction

## Getting Started

### Prerequisites

- Rust 2021 edition (for Rust development)
- Python 3.7+ (for Python bindings)
- Flask (for web interface)

### Installation

#### Rust Library

```bash
git clone https://github.com/nkkko/engramdb.git
cd engramdb
cargo build
```

#### Python Package

Install from PyPI:
```bash
pip install engramdb-py
```

Or build from source:
```bash
cd engramdb/python
pip install maturin
maturin develop
```

### Examples

EngramDB includes examples in both Rust and Python to help you get started.

#### Rust Examples

```bash
# Run examples using Cargo
cargo run --example basic_usage
cargo run --example memory_graph
cargo run --example memory_and_file_storage
```

#### Python Examples

```bash
# Run Python examples directly
cd examples/python
python basic_usage.py
python memory_graph.py
python agent_memory.py
```

Available examples demonstrate:
- Basic usage operations (create, save, retrieve memories)
- Working with in-memory and file-based storage
- Creating knowledge graphs with connected memories
- Building unified databases with different memory types
- Implementing agent memory systems

#### Web Interface

![EngramDB Web Interface](assets/engram_web_graph.jpg)

Start the web interface:
```bash
cd web
./run_web.sh  # Quick start script

# Or manual setup
cd web
python -m venv venv_app
source venv_app/bin/activate
pip install -r requirements.txt
python app_full.py
```

Access the interface at: http://localhost:8082

## Usage Examples

### Rust API

```rust
use engramdb::{Database, MemoryNode};
use engramdb::core::AttributeValue;

// Create an in-memory database
let mut db = Database::in_memory();

// Create a memory node
let mut memory = MemoryNode::new(vec![0.1, 0.2, 0.3, 0.4]);
memory.set_attribute("title".to_string(),
    AttributeValue::String("Important information".to_string()));

// Save to database
db.save(&memory)?;

// Search for similar memories
let query_vector = vec![0.15, 0.25, 0.35, 0.45];
let results = db.search_similar(&query_vector, 5, 0.0)?;

// Process results
for (memory_id, similarity) in &results {
    let node = db.load(*memory_id)?;
    println!("Found memory with similarity: {}", similarity);
}
```

### Python API

```python
import engramdb
import numpy as np

# Create an in-memory database
db = engramdb.Database.in_memory()

# Create a memory node with embeddings
embeddings = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
memory = engramdb.MemoryNode(embeddings)
memory.set_attribute("title", "Important information")
memory.set_attribute("importance", 0.8)

# Save to database
memory_id = db.save(memory)

# Search for similar memories
query_vector = np.array([0.15, 0.25, 0.35, 0.45], dtype=np.float32)
results = db.search_similar(query_vector, limit=5, threshold=0.0)

# Process results
for memory_id, similarity in results:
    memory = db.load(memory_id)
    print(f"Memory: {memory.get_attribute('title')}, Similarity: {similarity:.4f}")
```

## Architecture

EngramDB's architecture consists of:

1. **MemoryNode**: The fundamental storage unit combining:
   - Vector embeddings for semantic content
   - Graph connections to other memories
   - Temporal layers for versioning
   - Flexible key-value attributes

2. **Storage Engines**:
   - `MemoryStorageEngine`: Fast, in-memory storage
   - `FileStorageEngine`: Persistent file-based storage

3. **Vector Index**: Efficient similarity search

4. **Query System**: Flexible filters and constraints

5. **Language Bindings**:
   - Core implementation in Rust
   - Python bindings via PyO3

## Benchmarking

EngramDB includes a comprehensive benchmarking suite in the `benches/` directory for performance testing:

```bash
cd benches
python run_all_benchmarks.py  # Run all benchmarks
```

Available benchmarks measure:
- Embedding generation and storage performance
- Vector search efficiency
- Storage operations with varying database sizes
- Batch vs. individual operations
- In-memory vs. file-based storage performance

## Development

### Rust Development

```bash
cargo build
cargo test
cargo clippy
cargo fmt
```

### Python Development

```bash
cd python
pip install -r requirements-dev.txt
maturin develop
pytest tests/
```

### Web Development

```bash
cd web
python -m venv venv_app
source venv_app/bin/activate
pip install -r requirements.txt
python app_full.py
```

## License

EngramDB is licensed under the MIT License - see the LICENSE file for details.

**Note**: Documentation in the `docs/` directory and presentations in the `deck/` directory are © Copyright 2025, All Rights Reserved.

## Roadmap

Future enhancements for EngramDB include:

1. **Adaptive Embedding Indexing**: Self-optimizing index based on access patterns
2. **Predictive Memory Prefetching**: Anticipatory loading based on agent behavior
3. **Write-Optimized Memory Evolution**: Specialized structures for continuous updates
4. **Reflection-Optimized Query Engine**: Query language for agent introspection
5. **Cognitive-Inspired Storage Hierarchy**: Multi-tiered storage based on human memory models
6. **Distributed Storage Engine**: Support for clustered deployments