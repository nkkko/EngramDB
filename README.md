# EngramDB: Engram Database

EngramDB is a specialized database system designed specifically for agent memory management. It provides efficient storage, retrieval, and querying of agent memories using a unified memory representation model.

## Core Features

- **Unified Memory Representation**: Combines graph, vector, and temporal properties in a single primitive
- **Vector Similarity Search**: Find memories with similar semantic content
- **Flexible Storage Options**: In-memory database for testing and development, file-based for persistence
- **Query API**: Rich querying with vector similarity, attribute filters, and temporal constraints
- **Memory Evolution**: Track changes to memories over time with temporal layers

## Getting Started

### Prerequisites

- Rust 2021 edition
- Cargo

### Installation

Clone the repository and build the project:

```bash
git clone https://github.com/yourusername/engramdb.git
cd engramdb
cargo build
```

### Running Examples

EngramDB includes several examples that demonstrate different features:

```bash
# Run the basic usage example
cargo run --example basic_usage

# Run the memory graph example 
cargo run --example memory_graph

# Run the storage comparison example
cargo run --example memory_and_file_storage

# Run the unified database example
cargo run --example unified_database
```

## Usage

### Simple Usage

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

// Query with filters
let results = db.query()
    .with_vector(query_vector)
    .with_attribute_filter(/* ... */)
    .with_temporal_filter(/* ... */)
    .execute()?;
```

### Persistent Storage

```rust
use engramdb::{Database, DatabaseConfig};

// Create a persistent database
let config = DatabaseConfig {
    use_memory_storage: false,
    storage_dir: Some("./my_database".to_string()),
    cache_size: 100,
};

let mut db = Database::new(config)?;

// Or more simply
let mut db = Database::file_based("./my_database")?;

// Initialize to load existing memories into the vector index
db.initialize()?;
```

## Architecture

EngramDB is built around these core components:

1. **MemoryNode**: The fundamental unit of storage, combining:
   - Vector embeddings for semantic content
   - Graph connections to other memories
   - Temporal layers for versioning
   - Flexible attributes

2. **Storage Engines**:
   - `MemoryStorageEngine`: Fast, in-memory storage for testing
   - `FileStorageEngine`: Persistent file-based storage

3. **Vector Index**: For efficient similarity search

4. **Query System**: A fluent interface for building complex queries

5. **Database**: A unified interface combining all components

## Development

### Building

```bash
cargo build
```

### Testing

```bash
cargo test
```

### Linting

```bash
cargo clippy
```

### Formatting

```bash
cargo fmt
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Roadmap

Future enhancements planned for EngramDB:

1. **Adaptive Embedding Indexing**: Self-optimizing index based on access patterns
2. **Predictive Memory Prefetching**: Anticipatory loading based on agent behavior
3. **Write-Optimized Memory Evolution**: Specialized LSM tree for continuous updates
4. **Reflection-Optimized Query Engine**: Query language for agent introspection patterns
5. **Cognitive-Inspired Storage Hierarchy**: Multi-tiered storage based on human memory models