# Database API Reference

The `Database` is the main interface for interacting with EngramDB. This document provides a detailed reference for the `Database` API.

## Creating a Database

### `Database::in_memory()`

Creates a new in-memory database (volatile, for testing and development).

**Returns:**
- A new `Database` instance using memory storage

**Example:**
```rust
let mut db = Database::in_memory();
```

### `Database::file_based(dir)`

Creates a file-based database at the specified directory (persistent).

**Parameters:**
- `dir`: `&str` or `Path` - Path to the storage directory

**Returns:**
- A new `Database` instance using file storage, or an error if initialization failed

**Example:**
```rust
let mut db = Database::file_based("./my_database")?;
```

### `Database::new(config)`

Creates a new database with the given configuration.

**Parameters:**
- `config`: `DatabaseConfig` - Configuration options for the database

**Returns:**
- A new `Database` instance, or an error if initialization failed

**Example:**
```rust
let config = DatabaseConfig {
    use_memory_storage: false,
    storage_dir: Some("./my_database".to_string()),
    cache_size: 100,
};

let mut db = Database::new(config)?;
```

## DatabaseConfig

The `DatabaseConfig` struct provides configuration options for the database:

```rust
pub struct DatabaseConfig {
    pub use_memory_storage: bool,
    pub storage_dir: Option<String>,
    pub cache_size: usize,
}
```

**Fields:**
- `use_memory_storage`: Whether to use in-memory storage (volatile) or file-based storage (persistent)
- `storage_dir`: Directory path for file storage (ignored if using memory storage)
- `cache_size`: Size of the query result cache (0 to disable caching)

## Basic Operations

### `initialize()`

Initializes the database by loading existing memories into the vector index.

**Returns:**
- `Result<()>` - Success or an error

**Example:**
```rust
db.initialize()?;
```

### `save(node)`

Saves a memory node to the database.

**Parameters:**
- `node`: `&MemoryNode` - The memory node to save

**Returns:**
- `Result<Uuid>` - The ID of the saved memory node, or an error

**Example:**
```rust
let memory = MemoryNode::new(vec![0.1, 0.2, 0.3, 0.4]);
let memory_id = db.save(&memory)?;
```

### `load(id)`

Loads a memory node by its ID.

**Parameters:**
- `id`: `Uuid` - The ID of the memory node to load

**Returns:**
- `Result<MemoryNode>` - The loaded memory node, or an error if not found

**Example:**
```rust
let memory = db.load(memory_id)?;
```

### `delete(id)`

Deletes a memory node by its ID.

**Parameters:**
- `id`: `Uuid` - The ID of the memory node to delete

**Returns:**
- `Result<()>` - Success or an error

**Example:**
```rust
db.delete(memory_id)?;
```

### `list_all()`

Lists all memory node IDs in the database.

**Returns:**
- `Result<Vec<Uuid>>` - A vector of all memory node IDs, or an error

**Example:**
```rust
let all_ids = db.list_all()?;
println!("Database contains {} memories", all_ids.len());
```

## Vector Similarity Search

### `search_similar(query_vector, limit, threshold)`

Searches for memory nodes with similar vector embeddings.

**Parameters:**
- `query_vector`: `&[f32]` - The query vector
- `limit`: `usize` - Maximum number of results to return
- `threshold`: `f32` - Minimum similarity threshold (0.0 to 1.0)

**Returns:**
- `Result<Vec<(Uuid, f32)>>` - A vector of (ID, similarity) pairs, sorted by descending similarity

**Example:**
```rust
let query_vector = vec![0.15, 0.25, 0.35, 0.45];
let results = db.search_similar(&query_vector, 5, 0.0)?;

for (memory_id, similarity) in results {
    println!("Memory ID: {}, Similarity: {:.4}", memory_id, similarity);
}
```

## Query Builder

### `query()`

Creates a new query builder for complex queries.

**Returns:**
- A new `QueryBuilder` instance

**Example:**
```rust
let query_results = db.query()
    .with_vector(query_vector)
    .with_attribute_filter(category_filter)
    .with_attribute_filter(importance_filter)
    .execute()?;
```

## Connection Management

### `get_connections(id)`

Gets all connections for a memory node.

**Parameters:**
- `id`: `Uuid` - The ID of the memory node

**Returns:**
- `Result<Vec<ConnectionInfo>>` - A vector of connection information, or an error

**Example:**
```rust
let connections = db.get_connections(memory_id)?;
for connection in connections {
    println!("Connected to: {}, Type: {}, Strength: {:.2}",
        connection.target_id, connection.type_name, connection.strength);
}
```

### `add_connection(source_id, target_id, relationship_type, strength)`

Adds a connection between two memory nodes.

**Parameters:**
- `source_id`: `Uuid` - The ID of the source memory node
- `target_id`: `Uuid` - The ID of the target memory node
- `relationship_type`: `RelationshipType` - The type of relationship
- `strength`: `f32` - The strength of the connection (0.0 to 1.0)

**Returns:**
- `Result<()>` - Success or an error

**Example:**
```rust
db.add_connection(
    source_id,
    target_id,
    RelationshipType::Association,
    0.75
)?;
```

### `remove_connection(source_id, target_id)`

Removes a connection between two memory nodes.

**Parameters:**
- `source_id`: `Uuid` - The ID of the source memory node
- `target_id`: `Uuid` - The ID of the target memory node

**Returns:**
- `Result<bool>` - True if a connection was removed, false otherwise, or an error

**Example:**
```rust
let removed = db.remove_connection(source_id, target_id)?;
if removed {
    println!("Connection removed");
}
```

## Error Handling

The database operations return a `Result` type that can contain various error types:

```rust
pub enum EngramDbError {
    Storage(String),
    Query(String),
    Vector(String),
    Serialization(String),
    Validation(String),
    Other(String),
}
```

**Example:**
```rust
match db.load(non_existent_id) {
    Ok(memory) => println!("Found memory: {:?}", memory),
    Err(EngramDbError::Storage(msg)) => println!("Storage error: {}", msg),
    Err(e) => println!("Other error: {:?}", e),
}
```
