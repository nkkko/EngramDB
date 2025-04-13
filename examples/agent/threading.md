# EngramDB Threading Issues in Flask Application

## Issue Summary

When using EngramDB in a multi-threaded Flask application, we encounter thread safety issues that cause the application to crash. The error message indicates that EngramDB's Database object is not designed to be used across different threads.

## Error Message

```
assertion `left == right` failed: _engramdb::Database is unsendable, but sent to another thread!
  left: ThreadId(2)
 right: ThreadId(1)
```

## Technical Details

1. **Root Cause**: 
   The Rust implementation of EngramDB doesn't implement the `Send` trait for the `Database` class, which means it cannot be safely transferred between threads. When Flask serves multiple requests concurrently, it creates separate threads, each trying to access the same database object.

2. **Error Location**: 
   The panic occurs in the Pyo3 binding layer, specifically in `/Users/nikola/.cargo/registry/src/index.crates.io-6f17d22bba15001f/pyo3-0.19.2/src/impl_/pyclass.rs:927:9`
   
3. **Affected Code Paths**:
   - When initializing the agent context in a request handler
   - When accessing components stored in the database
   - During calls to `db.list_all()`, `db.load()`, and other database operations

4. **Attempted Workarounds**:
   - Setting Flask to run in single-threaded mode (`app.run(threaded=False, processes=1)`)
   - Using thread-local storage for the database connection
   - Moving database access to a dedicated thread

   None of these approaches fully resolved the issue due to how Flask and Werkzeug handle request processing.

## Detailed Error Stack Trace

```python
thread '<unnamed>' panicked at /Users/nikola/.cargo/registry/src/index.crates.io-6f17d22bba15001f/pyo3-0.19.2/src/impl_/pyclass.rs:927:9:
assertion `left == right` failed: _engramdb::Database is unsendable, but sent to another thread!
  left: ThreadId(2)
 right: ThreadId(1)

Exception in thread Thread-2 (process_request_thread):
Traceback (most recent call last):
  # ... [truncated stack trace] ...
  File "/flask_website_generator.py", line 1259, in home
    has_components = len(context.get_all_components()) > 0
                         ~~~~~~~~~~~~~~~~~~~~~~~~~~^^
  File "/flask_website_generator.py", line 388, in get_all_components
    memory_ids = self.db.list_all()
pyo3_runtime.PanicException: assertion `left == right` failed: _engramdb::Database is unsendable, but sent to another thread!
  left: ThreadId(2)
 right: ThreadId(1)
```

## Reproduction Steps

1. Create a Flask application that uses EngramDB
2. Initialize the database in the main thread
3. Define routes that access the database
4. Run the application with multiple threads (Flask default behavior)
5. Make concurrent requests to the application
6. Observe the panic errors when the second thread attempts to access the database

## Recommendations for EngramDB Developers

1. **Implement Thread Safety**:
   - Make `Database` implement the `Send` and `Sync` traits in Rust, which requires ensuring all fields are also `Send` and `Sync`
   - Or provide explicit documentation that the Database object is not thread-safe and should only be used in single-threaded contexts

2. **Connection Pool**:
   - Implement a thread-safe connection pool for database access
   - Allow each thread to obtain its own database connection

3. **Async Support**:
   - Consider adding async/await support which would allow non-blocking database operations
   - This would work well with async web frameworks like FastAPI

4. **Python GIL Considerations**:
   - Evaluate how the implementation interacts with Python's Global Interpreter Lock
   - Ensure proper handling of Python threads vs. Rust threads

5. **Documentation**:
   - Update documentation to clearly indicate threading limitations
   - Provide examples of safe usage patterns in multi-threaded web applications

## Temporary Workarounds

Until these issues are resolved, users have these options:

1. **Use Single-Threaded Mode**:
   ```python
   app.run(threaded=False, processes=1)
   ```
   However, this significantly limits concurrency.

2. **Create Per-Request Connections**:
   Initialize a new database connection for each request instead of reusing one.

3. **Implement a Request Queue**:
   Process all database operations in a single dedicated thread.

4. **Use Alternative Storage**:
   Replace EngramDB with a thread-safe alternative for web applications.

## Additional Context

The issue appears related to Pyo3's handling of Rust objects that don't implement `Send`. When Flask creates worker threads to handle requests, Pyo3 detects the thread ID mismatch and panics to prevent memory corruption or data races.