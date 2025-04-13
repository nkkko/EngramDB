"""
Example of using EngramDB with a Flask application in a thread-safe manner.

This example demonstrates how to use the ThreadSafeDatabase and ThreadSafeDatabasePool 
classes to safely use EngramDB in a multi-threaded Flask application.
"""

import os
import sys
import uuid
import threading
from pathlib import Path
from flask import Flask, jsonify, request, render_template_string

# Add the parent directory to the Python path to import engramdb_py
sys.path.append(str(Path(__file__).parent.parent))
import engramdb_py

# Create a ThreadSafeDatabasePool for connection management
DB_PATH = "thread_safe_test.engramdb"
try:
    # First, try creating a connection pool
    db_pool = engramdb_py.ThreadSafeDatabasePool.new(DB_PATH)
    print(f"Created thread-safe database pool at: {DB_PATH}")
except Exception as e:
    print(f"Could not create database pool: {e}")
    # Fall back to in-memory database
    db_pool = None
    print("Falling back to in-memory database for each request")

app = Flask(__name__)

@app.route('/')
def home():
    """Home page showing thread ID and database info"""
    thread_id = threading.get_ident()
    
    # Get a database connection
    if db_pool:
        db = db_pool.get_connection()
        db_type = "from pool"
    else:
        db = engramdb_py.ThreadSafeDatabase.in_memory()
        db_type = "in-memory"
    
    # Store a simple memory node
    node = engramdb_py.MemoryNode([0.1, 0.2, 0.3, 0.4, 0.5])
    node.set_attribute("thread_id", str(thread_id))
    node.set_attribute("timestamp", str(threading.current_thread().name))
    
    # Save to database
    memory_id = db.save(node)
    
    # Get total count
    try:
        memory_ids = db.list_all()
        count = len(memory_ids)
    except Exception as e:
        count = f"Error: {e}"
    
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>EngramDB Thread-Safe Example</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
            .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
            .info { background-color: #e3f2fd; }
            pre { background-color: #f5f5f5; padding: 10px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <h1>EngramDB Thread-Safe Example</h1>
        
        <div class="card info">
            <h2>Thread Information</h2>
            <p>Thread ID: {{ thread_id }}</p>
            <p>Thread Name: {{ thread_name }}</p>
            <p>Database Type: {{ db_type }}</p>
        </div>
        
        <div class="card">
            <h2>Memory Node</h2>
            <p>Created Memory ID: {{ memory_id }}</p>
            <p>Total Memories: {{ count }}</p>
        </div>
        
        <div class="card">
            <h2>Test Routes</h2>
            <ul>
                <li><a href="/memories">View All Memories</a></li>
                <li><a href="/concurrent-test">Test Concurrent Requests</a></li>
            </ul>
        </div>
    </body>
    </html>
    """, thread_id=thread_id, thread_name=threading.current_thread().name, 
         db_type=db_type, memory_id=memory_id, count=count)

@app.route('/memories')
def list_memories():
    """List all memories in the database"""
    # Get a database connection
    if db_pool:
        db = db_pool.get_connection()
    else:
        db = engramdb_py.ThreadSafeDatabase.in_memory()
    
    # Get all memories
    try:
        memory_ids = db.list_all()
        
        # Load each memory
        memories = []
        for memory_id in memory_ids:
            try:
                node = db.load(memory_id)
                thread_id = node.get_attribute("thread_id") or "Unknown"
                timestamp = node.get_attribute("timestamp") or "Unknown"
                
                memories.append({
                    "id": memory_id,
                    "thread_id": thread_id,
                    "timestamp": timestamp
                })
            except Exception as e:
                memories.append({
                    "id": memory_id,
                    "error": str(e)
                })
        
        return jsonify({
            "count": len(memories),
            "memories": memories
        })
    except Exception as e:
        return jsonify({
            "error": str(e)
        })

@app.route('/concurrent-test')
def concurrent_test():
    """Simulates concurrent access to the database"""
    import concurrent.futures
    import requests
    
    # Number of concurrent requests
    num_requests = 10
    base_url = request.host_url
    
    results = []
    
    # Use ThreadPoolExecutor to make concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
        futures = [executor.submit(requests.get, f"{base_url}concurrent-worker") for _ in range(num_requests)]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                response = future.result()
                results.append(response.json())
            except Exception as e:
                results.append({"error": str(e)})
    
    # Get the current memory count
    if db_pool:
        db = db_pool.get_connection()
    else:
        db = engramdb_py.ThreadSafeDatabase.in_memory()
    
    try:
        memory_count = len(db.list_all())
    except Exception as e:
        memory_count = f"Error: {e}"
    
    return jsonify({
        "success": True,
        "requests": num_requests,
        "completed": len(results),
        "memory_count": memory_count,
        "results": results
    })

@app.route('/concurrent-worker')
def concurrent_worker():
    """Worker endpoint for concurrent test"""
    thread_id = threading.get_ident()
    thread_name = threading.current_thread().name
    
    # Get a database connection
    if db_pool:
        db = db_pool.get_connection()
        db_type = "from pool"
    else:
        db = engramdb_py.ThreadSafeDatabase.in_memory()
        db_type = "in-memory"
    
    # Store a simple memory node
    node = engramdb_py.MemoryNode([0.1, 0.2, 0.3, 0.4, 0.5])
    node.set_attribute("thread_id", str(thread_id))
    node.set_attribute("thread_name", thread_name)
    node.set_attribute("worker", "true")
    
    # Save to database
    try:
        memory_id = db.save(node)
        success = True
        error = None
    except Exception as e:
        memory_id = None
        success = False
        error = str(e)
    
    return jsonify({
        "thread_id": thread_id,
        "thread_name": thread_name,
        "db_type": db_type,
        "memory_id": memory_id,
        "success": success,
        "error": error
    })

def clear_database():
    """Clear the database if it exists"""
    if Path(DB_PATH).exists():
        try:
            if db_pool:
                db = db_pool.get_connection()
                db.clear_all()
                print(f"Cleared existing database at {DB_PATH}")
            else:
                # If we don't have a pool, create a direct connection
                db = engramdb_py.ThreadSafeDatabase.file_based(DB_PATH)
                db.clear_all()
                print(f"Cleared existing database at {DB_PATH}")
        except Exception as e:
            print(f"Error clearing database: {e}")
            # If clearing fails, try to delete the database file
            try:
                os.remove(DB_PATH)
                print(f"Removed database file at {DB_PATH}")
            except Exception as remove_error:
                print(f"Error removing database file: {remove_error}")

if __name__ == '__main__':
    # Clear the database before starting
    clear_database()
    
    # Start the Flask app with threading enabled
    print("Starting Flask app with threading enabled...")
    print("Access the app at http://127.0.0.1:5000")
    app.run(debug=True, threaded=True, port=5000)