# TODO Style Guide

## Format Rules

- Use `[ ]` for incomplete tasks: `- [ ] Task description`
- Use `[x]` for completed tasks: `- [x] Task description`
- All tasks must have a priority marker: `[P0]`, `[P1]`, `[P2]`, or `[P3]`
  - `[P0]`: Critical - Must be done immediately
  - `[P1]`: High - Required for next milestone
  - `[P2]`: Medium - Important but not blocking
  - `[P3]`: Low - Nice to have

## Task Description Style
- Begin with an action verb in imperative form (e.g., "Add", "Implement", "Create")
- Be specific and concise
- Include acceptance criteria when appropriate
- For complex tasks, add sub-tasks with indentation

## Organization
- Group tasks by feature area or component
- Use headings (## for major sections, ### for subsections)
- Order tasks within sections by priority
- Move completed tasks to a "Completed" section with date

## Examples
```
- [x] [P1] Add basic vector search functionality (completed 2023-04-01)
- [ ] [P0] Fix memory leak in vector index when handling large embeddings
- [ ] [P1] Implement transaction support
  - [ ] Add begin_transaction() API
  - [ ] Add rollback functionality
  - [ ] Add commit functionality
```

---

# EngramDB Database Improvement Plan

This document outlines planned improvements to the EngramDB database system, inspired by SQLite's design principles and best practices in database engineering.

## High Priority Improvements

### 1. Transaction Support (ACID Properties)

- [ ] [P1] Implement atomic operations via transaction API
- [ ] [P1] Add journaling/write-ahead logging to prevent data corruption
- [ ] [P2] Support transaction isolation levels for concurrent access
- [ ] [P1] Enable commit/rollback functionality

```rust
// Example transaction API
let mut transaction = db.begin_transaction()?;
transaction.save(&memory1)?;
transaction.save(&memory2)?;
// Either commit all changes or roll back
transaction.commit()?; // Or transaction.rollback()?
```

### 2. Advanced Vector Indexing

- [x] [P0] Implement HNSW (Hierarchical Navigable Small World) algorithm for faster vector search (2025-04-10)
  - [x] [P0] Fix HNSW implementation to handle thread-safety requirements (2025-04-15)
  - [x] [P0] Optimize borrowing patterns in HNSW similarity search (2025-04-15)
- [x] [P1] Document HNSW implementation and usage in API reference (2025-04-11)
- [ ] [P1] Implement native multi-vector support in vector indexing without flattening
- [ ] [P1] Optimize memory usage in vector indices with deduplication and shared references
- [ ] [P2] Add IVF (Inverted File Index) support for large-scale vector collections
- [ ] [P2] Create product quantization for memory-efficient storage of vectors
- [ ] [P1] Add hybrid indexing that combines exact and approximate methods

```rust
// Configurable indexing
db.configure_index()
    .vector_algorithm(VectorAlgorithm::HNSW)
    .with_parameters(HNSWParams {
        m: 16,                 // Maximum number of connections per node
        ef_construction: 100,  // Size of dynamic candidate list during construction
        ef: 10,                // Size of dynamic candidate list during search
    })
    .build()?;
```

### 3. Thread Safety & Concurrency

- [ ] [P0] Implement thread safety in EngramDB Database class
  - [ ] Create ThreadSafeDatabase implementation using Arc and RwLock patterns
  - [ ] Make Database implement the Send and Sync traits in Rust
  - [ ] Ensure all internal fields are thread-safe with proper locking
  - [ ] Add tests for multi-threaded access patterns
  - [ ] Document thread safety guarantees
  - [ ] Replace misleading Database::clone with explicit ownership semantics
- [ ] [P1] Implement a connection pool for handling multiple clients
- [ ] [P0] Add read/write locks at different granularity levels
- [ ] [P1] Support concurrent readers with single writer pattern
- [ ] [P2] Implement deadlock detection and prevention

### 4. Query Optimization & Language

- [ ] [P2] Add statistics collection on data distribution
- [ ] [P1] Implement a query planner with cost-based optimization
- [ ] [P2] Create execution plans optimized for different query patterns
- [ ] [P1] Add query result caching for frequently accessed memories
- [ ] [P1] Design and implement a custom query language for memory retrieval
- [ ] [P2] Implement Okapi BM25 scoring for hybrid text relevance

```
# Example of potential query language:
FIND MEMORIES
SIMILAR TO [0.1, 0.3, 0.5, 0.2] WITH SIMILARITY > 0.7
WHERE attribute.category = "meeting" AND attribute.importance > 0.8
CREATED WITHIN LAST 7 DAYS
CONNECTED TO "d8f7a2e5-1c3b-4a6d-9e8f-5b7a2c3d1e0f" BY "Association"
LIMIT 10
```

## Medium Priority Improvements

### 5. Prepared Statements & Query Compilation

- [ ] [P2] Create a query compilation/preparation step
- [ ] [P2] Cache prepared queries to avoid repeated parsing
- [ ] [P2] Support parameterized queries for efficiency

```rust
// Example prepared query
let prepared_query = db.prepare_query()
    .with_vector_template(dimensions)
    .with_attribute_filter_template("category")
    .build()?;

// Reuse with different parameters
let results1 = prepared_query.execute(vec1, "work")?;
let results2 = prepared_query.execute(vec2, "personal")?;
```

### 6. Extensibility

- [ ] [P2] Create a plugin system for custom similarity functions
- [ ] [P3] Support user-defined attribute filters
- [ ] [P2] Allow custom storage backends
- [ ] [P3] Add JSON-LD support for enhanced semantic representation

```rust
// Register custom similarity function
db.register_similarity_function("manhattan_distance", |a, b| {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
});

// Example of using JSON-LD contexts
db.with_json_ld_context(r#"{
  "@context": {
    "engram": "http://example.org/engram/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "connections": {"@id": "engram:connections", "@container": "@set"},
    "knows": "foaf:knows"
  }
}"#);
```

### 7. Migration and Schema Evolution

- [ ] [P2] Support upgrading/downgrading database versions
- [ ] [P2] Add schema migration utilities
- [ ] [P3] Implement backward compatibility features
- [ ] [P2] Rename "MemoryNode" to "Engram" for better terminology alignment

```rust
// Example migration support
let migration = db.create_migration("v1-to-v2")
    .add_attribute("importance", AttributeValue::Float(0.5))
    .rename_attribute("type", "category")
    .apply_to_matching(AttributeFilter::equals("type", "memory"));

db.apply_migration(migration)?;

// Updated terminology example:
let engram = Engram::new(vec![0.1, 0.2, 0.3, 0.4]);
engram.set_attribute("title", "Meeting notes");
db.save(&engram)?;
```

## Lower Priority Improvements

### 8. Self-Contained Design

- [ ] [P3] Package database as a single dependency with no external requirements
- [ ] [P3] Ensure cross-platform compatibility
- [ ] [P2] Create zero-configuration defaults

### 9. Python SDK

- [x] [P1] Implement PyO3 bindings for core EngramDB functionality (completed 2025-04-01)
- [x] [P1] Create Pythonic wrapper classes for MemoryNode and Database (completed 2025-04-01)
- [ ] [P2] Develop Python-native query builder interface
- [ ] [P2] Write comprehensive Python documentation and examples
- [x] [P3] Add compatibility with popular Python ML libraries (numpy, pytorch, etc.) (2025-04-09)

### 9.1 Python SDK (engramdb-py) Enhancements

- [ ] [P1] Improve thread safety (critical fix)
  - [ ] Fix Send/Sync issues in the Rust/PyO3 layer so Database is inherently thread-safe
  - [ ] Make ThreadSafeDatabasePool the default and recommended way to instantiate the DB
  - [ ] Improve documentation explaining thread safety for web frameworks (Flask/FastAPI)
- [ ] [P1] Implement idiomatic Python API conventions
  - [ ] Use standard Python types (list, dict) for inputs/outputs where appropriate
  - [ ] Maintain support for NumPy arrays for efficient vector operations
  - [ ] Apply consistent naming and parameter conventions across the API
- [ ] [P1] Add first-class async support
  - [ ] Implement async versions of all I/O operations
  - [ ] Ensure compatibility with async frameworks (FastAPI, asyncio)
  - [ ] Optimize for LangChain compatibility which heavily relies on async
- [ ] [P2] Implement streaming for long-running operations
  - [ ] Add streaming search results
  - [ ] Support async iterators for large result sets
- [ ] [P2] Improve error handling
  - [ ] Use informative, standard Python exceptions
  - [ ] Add detailed error messages and context
  - [ ] Implement proper exception hierarchy
- [ ] [P2] Simplified connection methods
  - [ ] Support connection strings
  - [ ] Add environment variable configuration
  - [ ] Implement connection pooling with sensible defaults

### 10. Virtual Tables

- [ ] [P3] Support external data sources
- [ ] [P3] Create memory views that combine multiple queries
- [ ] [P3] Allow for schema-on-read approaches

### 12. Integration & API

- [ ] [P0] Implement example of using EngramDB with a simple agent using the pydantic.ai agentic framework
- [x] [P1] Implement RESTful API server using Rocket (2025-04-21)
  - [x] Implement database management endpoints (2025-04-21)
  - [x] Implement memory node CRUD operations (2025-04-21)
  - [x] Implement vector search endpoints (2025-04-21)
  - [x] Implement connection management (2025-04-21)
  - [x] Add authentication and authorization (2025-04-21)
  - [x] Set up OpenAPI documentation with Swagger UI (2025-04-21)
- [ ] [P2] Implement Model Context Protocol (MCP) server for LLM integration
- [ ] [P2] Create standardized API for RAG applications
- [x] [P1] Develop SDK libraries that use the REST API (2025-04-21)
  - [x] Create Python SDK (engramdb-client) (2025-04-21)
    - [x] Implement client class with API connection management (2025-04-21)
    - [x] Add database management operations (2025-04-21)
    - [x] Add memory node CRUD operations (2025-04-21)
    - [x] Add search and query capabilities (2025-04-21)
    - [x] Add connection management (2025-04-21)
    - [x] Add embedding operations (2025-04-21)
    - [x] Write comprehensive documentation and examples (2025-04-21)
  - [x] Create TypeScript SDK (@engramdb/client) (2025-04-21)
    - [x] Implement client class with API connection management (2025-04-21)
    - [x] Add database management operations (2025-04-21)
    - [x] Add memory node CRUD operations (2025-04-21)
    - [x] Add search and query capabilities (2025-04-21)
    - [x] Add connection management (2025-04-21)
    - [x] Add embedding operations (2025-04-21)
    - [x] Write comprehensive documentation and examples (2025-04-21)
  - [x] Ensure compatibility with existing PyO3 bindings API (2025-04-21)
- [ ] [P3] Implement multimodal memory support (images, audio, video)
- [ ] [P1] Add inter-agentic collaboration features for multi-agent orchestration

### 13. Business & Marketing

- [ ] [P1] Create a pitch deck for EngramDB targeting VC investors (15 main slides + 5 technical annexes)
- [ ] [P2] Develop comparison benchmarks against competing solutions
- [ ] [P3] Create case study examples of real-world applications

### 11. Testing Infrastructure

- [ ] [P2] Implement property-based testing for correctness
- [ ] [P3] Add fuzz testing to find edge cases
- [ ] [P1] Create a benchmark suite for performance regression testing
- [ ] [P1] Add comprehensive benchmarks to compare with SQLite for standard database operations
- [ ] [P2] Develop performance comparison metrics with SQLite for different workload patterns

### 14. CLI and Interface Improvements

- [x] [P1] Create basic CLI for EngramDB database management (2025-04-07)
- [ ] [P1] Improve storage engine error handling and robustness
- [ ] [P2] Standardize types across the API (e.g., consistent use of f32/f64)
- [ ] [P1] Refactor connection management
  - [ ] Extract common relationship type conversion logic to reduce code duplication
  - [ ] Enhance connection model to support bidirectional relationships more naturally
  - [ ] Implement proper methods for connection management in MemoryNode
- [ ] [P2] Add support for human-readable aliases or labels for memory nodes
- [ ] [P1] Extend query capabilities with filtering by multiple attributes
- [ ] [P2] Add optional schema validation for attributes
- [ ] [P2] Standardize behaviors across different storage engines
- [ ] [P1] Add support for bulk operations (import/export)
- [ ] [P3] Create a web interface for database management

## Implementation Phases

### Phase 1 (Short-term)
- All [P0] tasks:
  - ✓ Implement HNSW algorithm for faster vector search 
  - ✓ Fix thread-safety and borrowing patterns in HNSW implementation
  - Add read/write locks at different granularity levels
  - Create ThreadSafeDatabase implementation with Arc/RwLock patterns
  - Replace misleading Database::clone with explicit ownership semantics
  - Implement example using EngramDB with pydantic.ai agentic framework
- Critical [P1] tasks:
  - Transaction support (without isolation levels)
  - Read/write concurrency
  - Implement native multi-vector support without flattening
  - Optimize memory usage in vector indices
  - Refactor connection management and relationship handling
  - ✓ Implement RESTful API server using Rocket (2025-04-21)
  - ✓ Develop SDK libraries that use the REST API (2025-04-21)
  - Custom query language for memory retrieval
  - Improve storage engine error handling
  - Extend query capabilities
  - Add support for bulk operations

### Phase 2 (Mid-term)
- Remaining [P1] tasks
  - Improve thread safety in Python SDK (critical fix)
  - Implement idiomatic Python API conventions
  - Add first-class async support for Python SDK
  - Transaction support (without isolation levels)
  - Read/write concurrency
  - Implement native multi-vector support without flattening
  - Optimize memory usage in vector indices
  - Refactor connection management and relationship handling
- [P2] tasks for query optimization and performance
- Standardize types across API
- Enhance connection model
- Add support for human-readable aliases
- Implement optional schema validation
- Standardize behaviors across storage engines
- Implement helper functions for embeddings

### Phase 3 (Long-term)
- Remaining [P2] tasks
- [P3] tasks as needed
- Create a web interface for database management

## Completed Items

- [x] [P0] Create initial in-memory storage engine (2023-04-05)
- [x] [P0] Implement basic file storage persistence (2023-04-05)
- [x] [P0] Implement HNSW (Hierarchical Navigable Small World) algorithm for faster vector search (2025-04-10)
  - [x] [P0] Fix HNSW implementation to handle thread-safety requirements (2025-04-15)
  - [x] [P0] Optimize borrowing patterns in HNSW similarity search (2025-04-15)
  - [x] [P0] Clean up code warnings and improve error handling (2025-04-15)
- [x] [P1] Implement basic graph connections between memory nodes (2023-04-05)
- [x] [P1] Implement PyO3 bindings for core EngramDB functionality (2025-04-01)
- [x] [P1] Create Pythonic wrapper classes for MemoryNode and Database (2025-04-01)
- [x] [P1] Create basic CLI for EngramDB database management (2025-04-07)
- [x] [P1] Document HNSW implementation and usage in API reference (2025-04-11)
- [x] [P1] Implement RESTful API server using Rocket (2025-04-21)
  - [x] Implement database management endpoints (2025-04-21)
  - [x] Implement memory node CRUD operations (2025-04-21)
  - [x] Implement vector search endpoints (2025-04-21)
  - [x] Implement connection management (2025-04-21)
  - [x] Add authentication and authorization (2025-04-21)
  - [x] Set up OpenAPI documentation with Swagger UI (2025-04-21)
- [x] [P1] Develop SDK libraries that use the REST API (2025-04-21)
  - [x] Create Python SDK (engramdb-client) (2025-04-21)
  - [x] Create TypeScript SDK (@engramdb/client) (2025-04-21)
  - [x] Ensure compatibility with existing PyO3 bindings API (2025-04-21)
- [x] [P2] Implement helper functions for generating embeddings from text (2025-04-09)
- [x] [P2] Add support for multiple embedding models (2025-04-09)