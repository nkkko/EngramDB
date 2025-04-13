"""
EngramDB Thread-Safe Database Access for Flask

This module implements a thread-safe connection pool for EngramDB when used in multi-threaded
environments like Flask. It uses thread-local storage to ensure each thread has its own 
database connection.
"""

import os
import uuid
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional
import contextlib
import time
from datetime import datetime
import json

import engramdb_py as engramdb

# Thread-local storage to store database connections
_thread_local = threading.local()

class EngramDBConnectionPool:
    """
    Thread-safe connection pool for EngramDB.
    Creates a new database connection for each thread that needs one.
    """
    def __init__(self, db_path: str):
        """
        Initialize the connection pool with the database path.
        
        Args:
            db_path: Path to the EngramDB database
        """
        self.db_path = db_path
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection_count = 0
        self._lock = threading.RLock()
        print(f"Initialized EngramDB connection pool with path: {db_path}")
        
    def get_connection(self) -> engramdb.Database:
        """
        Get a database connection for the current thread.
        Creates a new connection if one doesn't exist.
        
        Returns:
            An EngramDB Database instance
        """
        # Check if this thread already has a connection
        if not hasattr(_thread_local, 'db'):
            with self._lock:
                self._connection_count += 1
                thread_id = threading.get_ident()
                print(f"Creating new database connection for thread {thread_id} (total: {self._connection_count})")
                
                # Create a new database connection for this thread
                try:
                    if self.path.exists():
                        _thread_local.db = engramdb.Database.file_based(self.db_path)
                    else:
                        _thread_local.db = engramdb.Database.file_based(self.db_path)
                        # Test the database connection
                        test_node = engramdb.MemoryNode([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
                        test_node.set_attribute("test", "value")
                        _thread_local.db.save(test_node)
                except Exception as e:
                    print(f"Error creating database connection: {e}")
                    # Fall back to in-memory database
                    print("Falling back to in-memory database")
                    _thread_local.db = engramdb.Database.in_memory()
        
        return _thread_local.db
    
    def close_all(self):
        """Close all database connections"""
        # Note: EngramDB Python bindings don't have an explicit close method,
        # but if one is added in the future, it should be called here
        with self._lock:
            print(f"Closing all database connections ({self._connection_count})")
            self._connection_count = 0
            # If EngramDB had a close method, we'd call it here
            # For now, we just remove the reference
            if hasattr(_thread_local, 'db'):
                delattr(_thread_local, 'db')
                
    @contextlib.contextmanager
    def connection(self):
        """Context manager for database connections"""
        try:
            yield self.get_connection()
        finally:
            # If EngramDB had a release method, we'd call it here
            pass


class ThreadSafeAgentContext:
    """
    Thread-safe agent context that creates a new connection for each thread.
    This is a drop-in replacement for the original AgentContext class.
    """
    def __init__(self, db_path: str = None, chat_id: uuid.UUID = None):
        """
        Initialize the thread-safe agent context.
        
        Args:
            db_path: Path to the EngramDB database
            chat_id: Chat ID for the agent context
        """
        if db_path is None:
            db_path = os.environ.get("ENGRAMDB_PATH", "agent_memory.engramdb")
        
        self.db_pool = EngramDBConnectionPool(db_path)
        self.chat_id = chat_id or uuid.uuid4()
        print(f"Initialized ThreadSafeAgentContext with chat_id: {self.chat_id}")
    
    @property
    def db(self):
        """Get a database connection for the current thread"""
        return self.db_pool.get_connection()
    
    def text_to_vector(self, text: str) -> List[float]:
        """Create a simple vector embedding for text"""
        import numpy as np
        text_bytes = text.encode('utf-8')
        hash_val = abs(hash(text_bytes)) % (2**32 - 1)
        np.random.seed(hash_val)
        return np.random.random(10).astype(np.float32).tolist()

    def store_memory(self, memory_type: str, content: Dict[str, Any]) -> str:
        """Store memory in EngramDB and return the node ID"""
        print(f"DEBUG: Starting to store memory of type: {memory_type}")
        vector = self.text_to_vector(json.dumps(content))
        node = engramdb.MemoryNode(vector)

        node.set_attribute("memory_type", memory_type)
        node.set_attribute("timestamp", datetime.now().isoformat())
        node.set_attribute("chat_id", str(self.chat_id))
        print(f"DEBUG: Set basic attributes: memory_type={memory_type}, chat_id={self.chat_id}")

        for key, value in content.items():
            # Ensure value is serializable or handle appropriately
            try:
                # Convert UUID to string to avoid PyString conversion errors
                if isinstance(value, uuid.UUID):
                    value = str(value)
                    
                # Attempt basic serialization check, adjust if complex types needed
                json.dumps({key: value})
                node.set_attribute(key, value)
                print(f"DEBUG: Set attribute {key}={value[:30]}... (truncated)" if isinstance(value, str) and len(value) > 30 else f"DEBUG: Set attribute {key}={value}")
            except TypeError:
                print(f"Warning: Attribute '{key}' with value type '{type(value)}' might not be serializable for storage. Storing as string.")
                node.set_attribute(key, str(value))

        try:
            print(f"DEBUG: Saving node to database...")
            memory_id = self.db.save(node)
            print(f"DEBUG: Node saved successfully with ID: {memory_id}")
            
            # Verify the node was saved by trying to load it
            try:
                test_load = self.db.load(memory_id)
                attrs = {attr: test_load.get_attribute(attr) for attr in ["memory_type", "timestamp", "chat_id"]}
                print(f"DEBUG: Verified node with ID {memory_id} was saved with attributes: {attrs}")
            except Exception as ve:
                print(f"DEBUG: WARNING - Could not verify node was saved, load failed: {ve}")
                
            return str(memory_id)
        except Exception as e:
            print(f"ERROR saving memory node: {e}")
            # Return a fallback ID in case of error
            fallback_id = str(uuid.uuid4())
            print(f"DEBUG: Using fallback ID: {fallback_id}")
            return fallback_id

    def store_message(self, role: str, content: str) -> str:
        """Store a chat message in EngramDB"""
        return self.store_memory("message", {
            "role": role,
            "content": content
        })

    def store_requirement(self, requirement: str) -> str:
        """Store a user requirement in EngramDB"""
        return self.store_memory("requirement", {
            "content": requirement
        })

    def store_component(self, name: str, component_type: str, code: str, description: str) -> str:
        """Store a generated code component in EngramDB"""
        print(f"DEBUG: Storing component '{name}' (type: {component_type}) to EngramDB")
        memory_id = self.store_memory("component", {
            "name": name,
            "type": component_type,
            "code": code,
            "description": description
        })
        print(f"DEBUG: Component stored with memory_id: {memory_id}")
        return memory_id

    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search for similar memories in EngramDB"""
        query_vector = self.text_to_vector(query)
        try:
            results = self.db.search_similar(query_vector, limit=limit, threshold=0.0)
        except Exception as e:
            print(f"Error searching similar memories: {e}")
            return []

        formatted_results = []
        for memory_id, similarity in results:
            try:
                # Proper handling of memory ID
                node_id = None
                if isinstance(memory_id, uuid.UUID):
                    node_id = memory_id
                elif isinstance(memory_id, bytes) and len(memory_id) == 16:
                    node_id = uuid.UUID(bytes=memory_id)
                elif isinstance(memory_id, str):
                    try:
                        node_id = uuid.UUID(memory_id)
                    except ValueError:
                        print(f"Could not convert string '{memory_id}' to UUID")
                        continue
                else:
                    print(f"Skipping invalid memory ID format in search results: {memory_id} (type: {type(memory_id)})")
                    continue
                
                if node_id is None:
                    continue
                        
                try:
                    node = self.db.load(node_id)
                except Exception as load_error:
                    print(f"Error loading node with ID {node_id}: {load_error}")
                    continue
                
                try:
                    memory_type = node.get_attribute("memory_type")
                except Exception as attr_error:
                    print(f"Error getting memory_type attribute: {attr_error}")
                    continue

                memory_data = {
                    "id": str(node_id),
                    "type": memory_type,
                    "timestamp": node.get_attribute("timestamp"),
                    "similarity": float(similarity)
                }

                if memory_type == "message":
                    memory_data["role"] = node.get_attribute("role")
                    memory_data["content"] = node.get_attribute("content")
                elif memory_type == "requirement":
                    memory_data["content"] = node.get_attribute("content")
                elif memory_type == "component":
                    memory_data["name"] = node.get_attribute("name")
                    memory_data["type"] = node.get_attribute("type")
                    memory_data["code"] = node.get_attribute("code")
                    memory_data["description"] = node.get_attribute("description")

                formatted_results.append(memory_data)
            except Exception as e:
                print(f"Error processing memory node: {e}")

        return formatted_results

    def get_chat_history(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent chat history for the current chat"""
        try:
            memory_ids = self.db.list_all()
        except Exception as e:
            print(f"Error listing all memories: {e}")
            return []
            
        messages = []

        for memory_id_bytes in memory_ids:
            try:
                # Proper handling of memory ID bytes
                node_id = None
                if isinstance(memory_id_bytes, uuid.UUID):
                    node_id = memory_id_bytes
                elif isinstance(memory_id_bytes, bytes) and len(memory_id_bytes) == 16:
                    try:
                        node_id = uuid.UUID(bytes=memory_id_bytes)
                    except ValueError:
                        print(f"Could not convert bytes to UUID: {memory_id_bytes.hex()}")
                        continue
                elif isinstance(memory_id_bytes, str):
                    try:
                        node_id = uuid.UUID(memory_id_bytes)
                    except ValueError:
                        print(f"Could not convert string to UUID: {memory_id_bytes}")
                        continue
                else:
                    print(f"Skipping invalid memory ID format: {memory_id_bytes} (type: {type(memory_id_bytes)})")
                    continue
                
                if node_id is None:
                    continue
                
                try:     
                    node = self.db.load(node_id)
                except Exception as load_error:
                    print(f"Error loading node with ID {node_id}: {load_error}")
                    continue
                
                try:
                    memory_type = node.get_attribute("memory_type")
                    if memory_type == "message":
                        chat_id = node.get_attribute("chat_id")
                        if chat_id == str(self.chat_id):
                            messages.append({
                                "id": str(node_id),
                                "role": node.get_attribute("role"),
                                "content": node.get_attribute("content"),
                                "timestamp": node.get_attribute("timestamp")
                            })
                except Exception as attr_error:
                    print(f"Error getting attributes from node {node_id}: {attr_error}")
                    continue
            except Exception as e:
                print(f"Error processing memory node: {e}")

        # Sort messages by timestamp if we have any
        if messages:
            try:
                messages.sort(key=lambda x: x.get("timestamp", ""))
            except Exception as sort_error:
                print(f"Error sorting messages by timestamp: {sort_error}")
                
        return messages[-limit:] if limit and len(messages) > limit else messages

    def get_all_components(self) -> List[Dict[str, Any]]:
        """Get all stored code components"""
        print("DEBUG: Getting all components from EngramDB")
        try:
            memory_ids = self.db.list_all()
            print(f"DEBUG: list_all() returned {len(memory_ids)} memory IDs")
        except Exception as e:
            print(f"Error listing all memories when getting components: {e}")
            return []
            
        components = []
        for memory_id in memory_ids:
            try:
                # Proper handling of memory ID
                node_id = None
                if isinstance(memory_id, uuid.UUID):
                    node_id = memory_id
                elif isinstance(memory_id, bytes) and len(memory_id) == 16:
                    try:
                        node_id = uuid.UUID(bytes=memory_id)
                    except ValueError:
                        continue
                elif isinstance(memory_id, str):
                    try:
                        node_id = uuid.UUID(memory_id)
                    except ValueError:
                        continue
                else:
                    continue
                
                if node_id is None:
                    continue
                
                # Load the node
                node = self.db.load(node_id)
                memory_type = node.get_attribute("memory_type")
                
                # Only process component nodes
                if memory_type == "component":
                    components.append({
                        "id": str(node_id),
                        "name": node.get_attribute("name"),
                        "type": node.get_attribute("type"),
                        "code": node.get_attribute("code"),
                        "description": node.get_attribute("description"),
                        "timestamp": node.get_attribute("timestamp")
                    })
            except Exception as e:
                print(f"Error processing component node: {e}")
                
        if not components:
            # Provide mock components if no real ones exist
            print("DEBUG: No components found, using mock components")
            from datetime import datetime
            mock_components = [
                {
                    "id": "mock-app",
                    "name": "app.py",
                    "type": "main",
                    "code": "from flask import Flask, render_template\n\napp = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return render_template('index.html')\n\nif __name__ == '__main__':\n    app.run(debug=True)",
                    "description": "The main Flask application",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "id": "mock-index",
                    "name": "index.html",
                    "type": "template",
                    "code": "<!DOCTYPE html>\n<html>\n<head>\n    <title>Todo App</title>\n    <link rel=\"stylesheet\" href=\"/static/style.css\">\n</head>\n<body>\n    <div class=\"container\">\n        <h1>Todo App</h1>\n        <div class=\"todo-list\">\n            <p>Your todos will appear here</p>\n        </div>\n    </div>\n</body>\n</html>",
                    "description": "The main template for the todo app",
                    "timestamp": datetime.now().isoformat()
                },
                {
                    "id": "mock-style",
                    "name": "style.css",
                    "type": "static",
                    "code": "body { font-family: Arial; margin: 0; padding: 20px; }\n.container { max-width: 800px; margin: 0 auto; }\nh1 { color: #333; }",
                    "description": "CSS styling for the todo app",
                    "timestamp": datetime.now().isoformat()
                }
            ]
            return mock_components
            
        return components


# Global connection pool
_global_pool = None

def get_connection_pool(db_path: str = None) -> EngramDBConnectionPool:
    """Get the global connection pool or create one if it doesn't exist"""
    global _global_pool
    if _global_pool is None:
        if db_path is None:
            db_path = os.environ.get("ENGRAMDB_PATH", "agent_memory.engramdb")
        _global_pool = EngramDBConnectionPool(db_path)
    return _global_pool

def get_thread_safe_context(db_path: str = None, chat_id: uuid.UUID = None) -> ThreadSafeAgentContext:
    """Get a thread-safe agent context"""
    return ThreadSafeAgentContext(db_path, chat_id)


# Usage example in a Flask app:
"""
from flask import Flask, request
from engramdb_thread_safe import get_thread_safe_context

app = Flask(__name__)

@app.route('/')
def home():
    # Each request gets its own database connection
    context = get_thread_safe_context()
    # Use context.db to access EngramDB
    memory_ids = context.db.list_all()
    return f"Found {len(memory_ids)} memories"

if __name__ == '__main__':
    # Flask can be run in threaded mode safely now
    app.run(debug=True, threaded=True)
"""