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

- [ ] [P0] Implement HNSW (Hierarchical Navigable Small World) algorithm for faster vector search
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

### 3. Connection Pooling & Concurrency

- [ ] [P1] Implement a connection pool for handling multiple clients
- [ ] [P0] Add read/write locks at different granularity levels
- [ ] [P1] Support concurrent readers with single writer pattern
- [ ] [P2] Implement deadlock detection and prevention

### 4. Query Optimization

- [ ] [P2] Add statistics collection on data distribution
- [ ] [P1] Implement a query planner with cost-based optimization
- [ ] [P2] Create execution plans optimized for different query patterns
- [ ] [P1] Add query result caching for frequently accessed memories

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

```rust
// Register custom similarity function
db.register_similarity_function("manhattan_distance", |a, b| {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
});
```

### 7. Migration and Schema Evolution

- [ ] [P2] Support upgrading/downgrading database versions
- [ ] [P2] Add schema migration utilities
- [ ] [P3] Implement backward compatibility features

```rust
// Example migration support
let migration = db.create_migration("v1-to-v2")
    .add_attribute("importance", AttributeValue::Float(0.5))
    .rename_attribute("type", "category")
    .apply_to_matching(AttributeFilter::equals("type", "memory"));

db.apply_migration(migration)?;
```

## Lower Priority Improvements

### 8. Self-Contained Design

- [ ] [P3] Package database as a single dependency with no external requirements
- [ ] [P3] Ensure cross-platform compatibility
- [ ] [P2] Create zero-configuration defaults

### 9. Python SDK

- [ ] [P1] Implement PyO3 bindings for core EngramDB functionality
- [ ] [P1] Create Pythonic wrapper classes for MemoryNode and Database
- [ ] [P2] Develop Python-native query builder interface
- [ ] [P2] Write comprehensive Python documentation and examples
- [ ] [P3] Add compatibility with popular Python ML libraries (numpy, pytorch, etc.)

### 10. Virtual Tables

- [ ] [P3] Support external data sources
- [ ] [P3] Create memory views that combine multiple queries
- [ ] [P3] Allow for schema-on-read approaches

### 11. Testing Infrastructure

- [ ] [P2] Implement property-based testing for correctness
- [ ] [P3] Add fuzz testing to find edge cases
- [ ] [P1] Create a benchmark suite for performance regression testing

## Implementation Phases

### Phase 1 (Short-term)
- All [P0] tasks
- Critical [P1] tasks:
  - Transaction support (without isolation levels)
  - Read/write concurrency
  - HNSW vector indexing

### Phase 2 (Mid-term)
- Remaining [P1] tasks
- [P2] tasks for query optimization and performance

### Phase 3 (Long-term)
- Remaining [P2] tasks
- [P3] tasks as needed

## Completed Items

- [x] [P0] Create initial in-memory storage engine (2023-04-05)
- [x] [P0] Implement basic file storage persistence (2023-04-05)