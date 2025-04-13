# EngramDB Thread Safety in Flask Applications

This document explains how to solve thread safety issues when using EngramDB with Flask or other multi-threaded web frameworks.

## Problem

EngramDB's Rust implementation doesn't implement the `Send` trait for the `Database` class, which means EngramDB Database objects cannot be safely transferred between threads. In a multi-threaded Flask application, each request runs in its own thread, leading to the following error:

```
assertion `left == right` failed: _engramdb::Database is unsendable, but sent to another thread!
  left: ThreadId(2)
 right: ThreadId(1)
```

## Solution

The solution involves creating a **thread-local connection pool** where each thread gets its own database connection. This implementation uses Python's `threading.local()` to store thread-specific database connections.

### Files

1. **engramdb_thread_safe.py**: Core implementation of the thread-safe EngramDB wrapper
2. **flask_website_generator_threadsafe.py**: Updated Flask website generator using the thread-safe implementation
3. **flask_threadsafe_demo.py**: Simple demo application that demonstrates thread safety

## How to Use

### 1. Basic Usage

```python
from flask import Flask, jsonify
from engramdb_thread_safe import get_thread_safe_context

app = Flask(__name__)

@app.route('/api/store')
def store_data():
    # Get a thread-safe context for this request
    context = get_thread_safe_context()
    
    # Use the context to access EngramDB
    memory_id = context.store_message("user", "Hello from Flask!")
    
    return jsonify({"memory_id": memory_id})

if __name__ == '__main__':
    # Can now run with threading=True safely
    app.run(debug=True, threaded=True)
```

### 2. Running the Demo

```bash
# Install requirements
pip install flask engramdb-py

# Run the demo application
python flask_threadsafe_demo.py
```

Visit http://localhost:5000 to see the demo in action.

### 3. Using with Existing Flask Applications

To convert an existing Flask application that uses EngramDB:

1. Import the thread-safe implementation:
   ```python
   from engramdb_thread_safe import get_thread_safe_context
   ```

2. Replace direct EngramDB database creation:
   ```python
   # Replace this:
   db = engramdb.Database.file_based("path/to/database")
   
   # With this:
   context = get_thread_safe_context("path/to/database")
   # Access db through context.db
   ```

3. Use the ThreadSafeAgentContext instead of a custom context class:
   ```python
   # Instead of your custom context class:
   # class MyContext:
   #     def __init__(self, db):
   #         self.db = db
   
   # Use the thread-safe implementation:
   from engramdb_thread_safe import ThreadSafeAgentContext
   ```

## How It Works

The implementation uses thread-local storage to create and maintain separate database connections for each thread:

1. **Thread-Local Storage**: We use `threading.local()` to store database connections specific to each thread.
2. **Connection Factory**: When a thread needs a database connection, we check if it already has one. If not, we create a new connection.
3. **Thread ID Tracking**: The thread ID is tracked to ensure each database connection is only used by its owner thread.

This approach allows Flask to operate in multi-threaded mode while safely using EngramDB.

## Technical Details

- Each request runs in its own thread and gets a separate database connection
- Database connections are cached on a per-thread basis for performance
- Any errors during database creation fall back to in-memory databases to prevent crashing
- The implementation provides complete API compatibility with the original EngramDB usage

## Performance Considerations

Creating a new database connection for each thread has some overhead, but it avoids the severe problem of thread-unsafe access. For most web applications, this overhead is acceptable, especially compared to the alternative of limiting Flask to single-threaded mode.

## Possible Future Improvements

This is a workaround until EngramDB's Rust implementation is updated to implement the `Send` trait. Long-term improvements could include:

1. Making the `Database` class in EngramDB implement `Send` and `Sync` traits
2. Adding proper connection pooling to EngramDB's Python bindings
3. Implementing async/await support for non-blocking database operations

## Running the Thread-Safe Website Generator

```bash
python flask_website_generator_threadsafe.py web
```

This runs the Flask website generator with threading enabled, allowing multiple concurrent requests.