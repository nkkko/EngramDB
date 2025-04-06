# Query API Reference

The Query API provides a flexible way to search for memories based on multiple criteria. This document provides a detailed reference for the query-related APIs in EngramDB.

## QueryBuilder

The `QueryBuilder` class provides a fluent interface for building complex queries.

### Creating a Query

#### `QueryBuilder::new()`

Creates a new query builder.

**Returns:**
- A new `QueryBuilder` instance

**Example:**
```rust
let query = QueryBuilder::new();
```

### Query Criteria

#### `with_vector(query_vector)`

Sets the query vector for similarity search.

**Parameters:**
- `query_vector`: `Vec<f32>` - The query vector

**Returns:**
- `Self` - The query builder for method chaining

**Example:**
```rust
let query = QueryBuilder::new()
    .with_vector(vec![0.1, 0.2, 0.3, 0.4]);
```

#### `with_similarity_threshold(threshold)`

Sets the minimum similarity threshold for vector search.

**Parameters:**
- `threshold`: `f32` - Minimum similarity threshold (0.0 to 1.0)

**Returns:**
- `Self` - The query builder for method chaining

**Example:**
```rust
let query = QueryBuilder::new()
    .with_vector(vec![0.1, 0.2, 0.3, 0.4])
    .with_similarity_threshold(0.7);
```

#### `with_attribute_filter(filter)`

Adds an attribute filter to the query.

**Parameters:**
- `filter`: `AttributeFilter` - The attribute filter to add

**Returns:**
- `Self` - The query builder for method chaining

**Example:**
```rust
let category_filter = AttributeFilter::equals("category", "work");
let importance_filter = AttributeFilter::greater_than("importance", 0.7);

let query = QueryBuilder::new()
    .with_attribute_filter(category_filter)
    .with_attribute_filter(importance_filter);
```

#### `with_temporal_filter(filter)`

Adds a temporal filter to the query.

**Parameters:**
- `filter`: `TemporalFilter` - The temporal filter to add

**Returns:**
- `Self` - The query builder for method chaining

**Example:**
```rust
let time_filter = TemporalFilter::within_last(24 * 60 * 60); // Last 24 hours

let query = QueryBuilder::new()
    .with_temporal_filter(time_filter);
```

#### `with_limit(limit)`

Sets the maximum number of results to return.

**Parameters:**
- `limit`: `usize` - Maximum number of results

**Returns:**
- `Self` - The query builder for method chaining

**Example:**
```rust
let query = QueryBuilder::new()
    .with_limit(10);
```

#### `with_exclude_ids(ids)`

Adds IDs to exclude from the results.

**Parameters:**
- `ids`: `Vec<Uuid>` - IDs to exclude

**Returns:**
- `Self` - The query builder for method chaining

**Example:**
```rust
let query = QueryBuilder::new()
    .with_exclude_ids(vec![id1, id2, id3]);
```

### Executing the Query

#### `execute(vector_index, memory_nodes)`

Executes the query against a vector index and a set of memory nodes.

**Parameters:**
- `vector_index`: `&VectorIndex` - The vector index to search
- `memory_nodes`: `F` - A function that can retrieve memory nodes by ID

**Returns:**
- `Result<Vec<MemoryNode>>` - A vector of memory nodes matching the query, sorted by relevance

**Example:**
```rust
// Using the Database's query method (recommended)
let results = db.query()
    .with_vector(query_vector)
    .with_attribute_filter(category_filter)
    .execute()?;

// Manual execution (advanced)
let results = query.execute(&vector_index, |id| storage.load(id))?;
```

## AttributeFilter

The `AttributeFilter` class provides methods for filtering memories based on their attributes.

### Creating Filters

#### `AttributeFilter::equals(key, value)`

Creates a filter that matches attributes equal to the specified value.

**Parameters:**
- `key`: `&str` - The attribute key
- `value`: `AttributeValue` - The value to compare against

**Returns:**
- A new `AttributeFilter` instance

**Example:**
```rust
let filter = AttributeFilter::equals("category", AttributeValue::String("work".to_string()));
```

#### `AttributeFilter::not_equals(key, value)`

Creates a filter that matches attributes not equal to the specified value.

**Parameters:**
- `key`: `&str` - The attribute key
- `value`: `AttributeValue` - The value to compare against

**Returns:**
- A new `AttributeFilter` instance

**Example:**
```rust
let filter = AttributeFilter::not_equals("category", AttributeValue::String("personal".to_string()));
```

#### `AttributeFilter::greater_than(key, value)`

Creates a filter that matches numeric attributes greater than the specified value.

**Parameters:**
- `key`: `&str` - The attribute key
- `value`: `AttributeValue` - The value to compare against (must be numeric)

**Returns:**
- A new `AttributeFilter` instance

**Example:**
```rust
let filter = AttributeFilter::greater_than("importance", AttributeValue::Float(0.7));
```

#### `AttributeFilter::less_than(key, value)`

Creates a filter that matches numeric attributes less than the specified value.

**Parameters:**
- `key`: `&str` - The attribute key
- `value`: `AttributeValue` - The value to compare against (must be numeric)

**Returns:**
- A new `AttributeFilter` instance

**Example:**
```rust
let filter = AttributeFilter::less_than("priority", AttributeValue::Integer(3));
```

#### `AttributeFilter::contains(key, substring)`

Creates a filter that matches string attributes containing the specified substring.

**Parameters:**
- `key`: `&str` - The attribute key
- `substring`: `&str` - The substring to search for

**Returns:**
- A new `AttributeFilter` instance

**Example:**
```rust
let filter = AttributeFilter::contains("title", "meeting");
```

#### `AttributeFilter::exists(key)`

Creates a filter that matches memories where the specified attribute exists.

**Parameters:**
- `key`: `&str` - The attribute key

**Returns:**
- A new `AttributeFilter` instance

**Example:**
```rust
let filter = AttributeFilter::exists("category");
```

#### `AttributeFilter::not_exists(key)`

Creates a filter that matches memories where the specified attribute does not exist.

**Parameters:**
- `key`: `&str` - The attribute key

**Returns:**
- A new `AttributeFilter` instance

**Example:**
```rust
let filter = AttributeFilter::not_exists("category");
```

## TemporalFilter

The `TemporalFilter` class provides methods for filtering memories based on their temporal properties.

### Creating Filters

#### `TemporalFilter::before(timestamp)`

Creates a filter that matches memories created before the specified timestamp.

**Parameters:**
- `timestamp`: `u64` - The Unix timestamp

**Returns:**
- A new `TemporalFilter` instance

**Example:**
```rust
let filter = TemporalFilter::before(1609459200); // Before January 1, 2021
```

#### `TemporalFilter::after(timestamp)`

Creates a filter that matches memories created after the specified timestamp.

**Parameters:**
- `timestamp`: `u64` - The Unix timestamp

**Returns:**
- A new `TemporalFilter` instance

**Example:**
```rust
let filter = TemporalFilter::after(1609459200); // After January 1, 2021
```

#### `TemporalFilter::between(start_timestamp, end_timestamp)`

Creates a filter that matches memories created between the specified timestamps.

**Parameters:**
- `start_timestamp`: `u64` - The start Unix timestamp
- `end_timestamp`: `u64` - The end Unix timestamp

**Returns:**
- A new `TemporalFilter` instance

**Example:**
```rust
let filter = TemporalFilter::between(1609459200, 1640995200); // Between Jan 1, 2021 and Jan 1, 2022
```

#### `TemporalFilter::within_last(seconds)`

Creates a filter that matches memories created within the last specified number of seconds.

**Parameters:**
- `seconds`: `u64` - Number of seconds

**Returns:**
- A new `TemporalFilter` instance

**Example:**
```rust
let filter = TemporalFilter::within_last(24 * 60 * 60); // Within the last 24 hours
```

## Combined Query Example

Here's an example of a complex query that combines multiple criteria:

```rust
// Create filters
let category_filter = AttributeFilter::equals(
    "category", 
    AttributeValue::String("work".to_string())
);

let importance_filter = AttributeFilter::greater_than(
    "importance", 
    AttributeValue::Float(0.7)
);

let time_filter = TemporalFilter::within_last(7 * 24 * 60 * 60); // Last week

// Build and execute query
let results = db.query()
    .with_vector(vec![0.1, 0.2, 0.3, 0.4])
    .with_similarity_threshold(0.6)
    .with_attribute_filter(category_filter)
    .with_attribute_filter(importance_filter)
    .with_temporal_filter(time_filter)
    .with_limit(10)
    .execute()?;

// Process results
for memory in results {
    println!("Found memory: {:?}", memory);
}
```
