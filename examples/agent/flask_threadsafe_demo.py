"""
Demo script to demonstrate the thread-safe EngramDB implementation with Flask.

This script sets up a simple Flask application that uses EngramDB in a multi-threaded environment.
"""

import os
import sys
import time
import threading
import uuid
from flask import Flask, jsonify, request, render_template_string

# Import our thread-safe EngramDB implementation
from engramdb_thread_safe import get_thread_safe_context, ThreadSafeAgentContext

app = Flask(__name__)

# Get a thread-safe agent context
def get_context():
    return get_thread_safe_context()

@app.route('/')
def home():
    """Home page with information about the demo"""
    thread_id = threading.get_ident()
    context = get_context()
    
    # Store a message for this request
    memory_id = context.store_message("system", f"Home page accessed from thread {thread_id}")
    
    # Get the total number of messages
    try:
        memories = context.db.list_all()
        memory_count = len(memories)
    except Exception as e:
        memory_count = f"Error listing memories: {e}"
    
    return render_template_string('''
        <html>
        <head>
            <title>EngramDB Thread-Safe Demo</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
                .info { background: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; margin-bottom: 20px; }
                pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
                .btn { display: inline-block; padding: 10px 15px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; }
            </style>
        </head>
        <body>
            <h1>EngramDB Thread-Safe Demo</h1>
            
            <div class="info">
                <h2>Thread-Safe EngramDB Demo</h2>
                <p>This demo shows EngramDB working in a multi-threaded Flask application.</p>
                <p>Current thread ID: {{ thread_id }}</p>
                <p>Memory ID for this request: {{ memory_id }}</p>
                <p>Total memories in database: {{ memory_count }}</p>
            </div>
            
            <div class="card">
                <h3>Available Test Endpoints:</h3>
                <ul>
                    <li><a href="/store">Store a random message</a></li>
                    <li><a href="/messages">View stored messages</a></li>
                    <li><a href="/test-concurrent">Test 10 concurrent requests</a></li>
                </ul>
            </div>
            
            <div class="card">
                <h3>How It Works:</h3>
                <p>Each Flask request runs in its own thread. The thread-safe EngramDB implementation creates a new database connection for each thread using thread-local storage.</p>
                <pre>
# Example of how to use the thread-safe implementation:
from engramdb_thread_safe import get_thread_safe_context

@app.route('/api/endpoint')
def api_endpoint():
    # Each request gets its own thread-safe context
    context = get_thread_safe_context()
    
    # Use context.db to access EngramDB operations
    memory_id = context.store_message("user", "Hello, EngramDB!")
    
    return {"success": True, "memory_id": memory_id}
                </pre>
            </div>
        </body>
        </html>
    ''', thread_id=thread_id, memory_id=memory_id, memory_count=memory_count)

@app.route('/store')
def store_message():
    """Store a random message in EngramDB"""
    thread_id = threading.get_ident()
    context = get_context()
    
    # Generate a random message
    message = f"Test message from thread {thread_id} at {time.time()}"
    
    # Store the message
    memory_id = context.store_message("system", message)
    
    return jsonify({
        "success": True,
        "thread_id": thread_id,
        "message": message,
        "memory_id": memory_id
    })

@app.route('/messages')
def list_messages():
    """List all messages in EngramDB"""
    thread_id = threading.get_ident()
    context = get_context()
    
    # Get all memories and filter for messages
    memory_ids = context.db.list_all()
    
    messages = []
    for memory_id in memory_ids:
        try:
            # Convert memory_id to UUID if needed
            if isinstance(memory_id, bytes):
                memory_id = uuid.UUID(bytes=memory_id)
            elif isinstance(memory_id, str):
                memory_id = uuid.UUID(memory_id)
                
            node = context.db.load(memory_id)
            memory_type = node.get_attribute("memory_type")
            
            if memory_type == "message":
                messages.append({
                    "id": str(memory_id),
                    "role": node.get_attribute("role"),
                    "content": node.get_attribute("content"),
                    "timestamp": node.get_attribute("timestamp")
                })
        except Exception as e:
            print(f"Error loading memory {memory_id}: {e}")
    
    # Sort messages by timestamp if available
    messages.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    
    return render_template_string('''
        <html>
        <head>
            <title>Stored Messages</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
                .message { background: #f9f9f9; padding: 10px; margin-bottom: 10px; border-radius: 5px; }
                .system { border-left: 5px solid #4CAF50; }
                .user { border-left: 5px solid #2196F3; }
                .assistant { border-left: 5px solid #FF9800; }
                .timestamp { color: #999; font-size: 0.8em; }
            </style>
        </head>
        <body>
            <h1>Stored Messages</h1>
            <p>Current thread ID: {{ thread_id }}</p>
            <p>Found {{ messages|length }} messages in the database.</p>
            
            <div class="card">
                <h3>Messages:</h3>
                {% for message in messages %}
                <div class="message {{ message.role }}">
                    <strong>{{ message.role|upper }}:</strong> {{ message.content }}
                    <div class="timestamp">ID: {{ message.id }} | {{ message.timestamp }}</div>
                </div>
                {% endfor %}
            </div>
            
            <p><a href="/">Back to Home</a></p>
        </body>
        </html>
    ''', thread_id=thread_id, messages=messages)

@app.route('/test-concurrent')
def test_concurrent():
    """Test 10 concurrent requests to ensure thread safety"""
    import requests
    import concurrent.futures
    
    base_url = request.url_root
    
    results = []
    
    # Use ThreadPoolExecutor to make concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # Submit 10 requests to the /store endpoint
        futures = [executor.submit(requests.get, f"{base_url}store") for _ in range(10)]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                response = future.result()
                results.append(response.json())
            except Exception as e:
                results.append({"error": str(e)})
    
    # Check for any errors
    errors = [r for r in results if "error" in r]
    
    return render_template_string('''
        <html>
        <head>
            <title>Concurrent Test Results</title>
            <style>
                body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
                .success { background: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; }
                .error { background: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336; }
                pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
            </style>
        </head>
        <body>
            <h1>Concurrent Request Test Results</h1>
            
            {% if errors %}
            <div class="error">
                <h3>Test Result: FAILED</h3>
                <p>Found {{ errors|length }} errors in {{ results|length }} concurrent requests.</p>
            </div>
            {% else %}
            <div class="success">
                <h3>Test Result: PASSED</h3>
                <p>Successfully completed {{ results|length }} concurrent requests with no errors!</p>
            </div>
            {% endif %}
            
            <div class="card">
                <h3>Request Results:</h3>
                <pre>{{ results_json }}</pre>
            </div>
            
            <p><a href="/">Back to Home</a> | <a href="/messages">View All Messages</a></p>
        </body>
        </html>
    ''', results=results, errors=errors, results_json=str(results))

if __name__ == "__main__":
    host = '0.0.0.0'
    port = 5000
    
    print(f"Starting Flask thread-safe EngramDB demo on http://{host}:{port}")
    print("Press Ctrl+C to exit")
    
    # Run with threading=True to demonstrate thread safety
    app.run(host=host, port=port, debug=True, threaded=True)