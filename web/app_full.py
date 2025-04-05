import os
import sys
import json
import time
import traceback

# Try to import numpy, but continue if not available
try:
    import numpy as np
except ImportError:
    # Create a minimal numpy-like module for our needs
    class NumpyMock:
        def array(self, list_data):
            return list_data
    np = NumpyMock()
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.exceptions import NotFound
import uuid
import logging

# Import our embedding utilities
try:
    from embedding_utils import (
        generate_embedding_for_memory,
        generate_embedding_from_query,
        mock_embeddings
    )
    EMBEDDING_MODEL_AVAILABLE = True
    print("Embedding model utilities available")
except Exception as e:
    print(f"Warning: Embedding model utilities not available: {e}")
    print(traceback.format_exc())
    EMBEDDING_MODEL_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Helper functions to handle methods that require Python GIL
def get_embeddings_as_list(memory_node):
    """
    Safely get embeddings from a memory node as a Python list.
    The get_embeddings method requires the Python GIL parameter, which is auto-managed
    when called directly from Python (not via properties or in templates).
    """
    try:
        # Get embeddings
        embeddings = memory_node.get_embeddings()
        # Try to convert to list if it's a numpy array
        try:
            return embeddings.tolist()
        except AttributeError:
            # If it's already a list or our mock implementation, just return it
            return embeddings if isinstance(embeddings, list) else list(embeddings)
    except Exception as e:
        logger.error(f"Error getting embeddings: {e}")
        return []
        
def get_attributes_as_dict(memory_node):
    """
    Safely get attributes from a memory node as a Python dictionary.
    The attributes method requires the Python GIL parameter.
    """
    try:
        attributes = memory_node.attributes()
        return dict(attributes)
    except Exception as e:
        logger.error(f"Error getting attributes: {e}")
        return {}

# Add parent directory to sys.path to find engramdb
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Create a mock MemoryNode class for demonstration if engramdb_py is not available
class MockMemoryNode:
    def __init__(self, embeddings=None):
        self.id = str(uuid.uuid4())
        self._embeddings = embeddings or [0.1, 0.2, 0.3, 0.4]
        self._attributes = {}
    
    def get_embeddings(self):
        return np.array(self._embeddings)
    
    def set_embeddings(self, embeddings):
        self._embeddings = embeddings
    
    def get_attribute(self, key):
        return self._attributes.get(key)
    
    def set_attribute(self, key, value):
        self._attributes[key] = value
    
    def attributes(self):
        return self._attributes.items()

# Create a mock Database class for demonstration
class MockDatabase:
    def __init__(self):
        self.memories = {}
    
    @classmethod
    def file_based(cls, path):
        return cls()
    
    @classmethod
    def in_memory(cls):
        return cls()
    
    def save(self, memory):
        self.memories[memory.id] = memory
        return memory.id
    
    def load(self, memory_id):
        if memory_id not in self.memories:
            raise ValueError(f"Memory with ID {memory_id} not found")
        return self.memories[memory_id]
    
    def delete(self, memory_id):
        if memory_id in self.memories:
            del self.memories[memory_id]
    
    def list_all(self):
        return list(self.memories.keys())
    
    def search_similar(self, query_vector, limit=10, threshold=0.0):
        # Simple mock for vector search - just return random similarities
        import random
        results = []
        for memory_id, memory in self.memories.items():
            # Skip if embedding dimensions don't match
            if len(memory._embeddings) != len(query_vector):
                continue
            
            # Calculate a random similarity score
            similarity = random.uniform(threshold, 1.0)
            results.append((memory_id, similarity))
        
        # Sort by similarity (highest first) and return top limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

try:
    # Try to import engramdb
    from engramdb_py import MemoryNode, Database, RelationshipType
    logger.info("Successfully imported engramdb_py")
    # Test the MemoryNode to verify methods
    test_node = MemoryNode([0.1, 0.2])
    logger.info(f"Created test node with ID: {test_node.id}")
except ImportError as e:
    logger.warning(f"Using mock implementation: {e}")
    # Use mock implementations
    MemoryNode = MockMemoryNode
    Database = MockDatabase
    
    # Create a simple RelationshipType enum for mocks
    class RelationshipType:
        PREDECESSOR = "predecessor"
        SUCCESSOR = "successor"
        REFERENCE = "reference"

app = Flask(__name__)
app.secret_key = 'engramdb-secret-key'  # For flash messages

# Enable verbose Flask logging
app.logger.setLevel(logging.DEBUG)

# Create a database instance
DB_PATH = os.path.join(os.path.dirname(__file__), "engramdb_data")
os.makedirs(DB_PATH, exist_ok=True)

try:
    db = Database.file_based(DB_PATH)
    print(f"EngramDB initialized at {DB_PATH}")
    
    # Check if the database is empty
    ids = db.list_all()
    print(f"Found {len(ids)} existing memories")
except Exception as e:
    print(f"Error initializing database: {e}")
    db = Database.in_memory()
    print("Fallback to in-memory database")

@app.route('/')
def index():
    """Homepage showing the list of all memories."""
    try:
        memory_ids = db.list_all()
        memories = []
        
        for memory_id in memory_ids:
            memory = db.load(memory_id)
            # Extract important attributes for display
            memory_data = {
                'id': memory_id,
                'title': memory.get_attribute('title') or 'Untitled',
                'category': memory.get_attribute('category') or 'Uncategorized',
                'importance': memory.get_attribute('importance') or 0.0,
                'embedding_size': len(get_embeddings_as_list(memory)),
            }
            memories.append(memory_data)
            
        return render_template('index.html', memories=memories)
    except Exception as e:
        flash(f"Error loading memories: {str(e)}", 'error')
        return render_template('index.html', memories=[])

@app.route('/memory/new', methods=['GET', 'POST'])
def create_memory():
    """Create a new memory."""
    if request.method == 'POST':
        try:
            title = request.form.get('title', 'Untitled')
            category = request.form.get('category', 'Uncategorized')
            importance = float(request.form.get('importance', 0.5))
            memory_content = request.form.get('content', '')
            use_model_embeddings = request.form.get('use_model_embeddings') == 'true'
            
            # Get embeddings either from form or generate from content
            embeddings = None
            
            if use_model_embeddings and memory_content and EMBEDDING_MODEL_AVAILABLE:
                logger.info(f"Generating embeddings from content: '{memory_content[:50]}...'")
                embeddings = generate_embedding_for_memory(memory_content, category)
                if embeddings is not None:
                    logger.info(f"Generated embeddings of shape {embeddings.shape}")
                    # Convert numpy array to list to avoid ambiguity error
                    embeddings = embeddings.tolist()
                else:
                    logger.warning("Failed to generate embeddings from model")
                    # Fallback to random embeddings
                    embeddings = mock_embeddings().tolist()
                    flash("Could not generate embeddings from model. Using random embeddings instead.", "warning")
            else:
                # Parse embeddings from form
                embedding_text = request.form.get('embeddings', '')
                try:
                    # Try to parse as JSON array
                    embeddings = json.loads(embedding_text)
                    if not isinstance(embeddings, list):
                        raise ValueError("Embeddings must be a list of numbers")
                except json.JSONDecodeError:
                    # Fallback to parsing as space-separated numbers
                    embeddings = [float(x) for x in embedding_text.split()]
                
            if not embeddings:
                # Default embedding if none provided
                embeddings = [0.1, 0.2, 0.3, 0.4]
            
            # Create the memory node
            memory = MemoryNode(embeddings)
            memory.set_attribute('title', title)
            memory.set_attribute('category', category)
            memory.set_attribute('importance', importance)
            
            # Store the content if provided
            if memory_content:
                memory.set_attribute('content', memory_content)
            
            # Save to database
            memory_id = db.save(memory)
            
            flash(f"Memory '{title}' created successfully!", 'success')
            return redirect(url_for('view_memory', memory_id=memory_id))
        except Exception as e:
            flash(f"Error creating memory: {str(e)}", 'error')
            
    # Pass whether the embedding model is available to the template
    return render_template('create.html', embedding_model_available=EMBEDDING_MODEL_AVAILABLE)

@app.route('/memory/<memory_id>')
def view_memory(memory_id):
    """View a single memory."""
    try:
        memory = db.load(memory_id)
        
        # Prepare data for template
        memory_data = {
            'id': memory_id,
            'title': memory.get_attribute('title') or 'Untitled',
            'category': memory.get_attribute('category') or 'Uncategorized',
            'importance': memory.get_attribute('importance') or 0.0,
            'embeddings': get_embeddings_as_list(memory),
            'attributes': {}
        }
        
        # Get all attributes
        attributes = get_attributes_as_dict(memory)
        for key, value in attributes.items():
            if key not in ['title', 'category', 'importance']:
                memory_data['attributes'][key] = value
                
        return render_template('view.html', memory=memory_data)
    except Exception as e:
        flash(f"Error loading memory: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/memory/<memory_id>/edit', methods=['GET', 'POST'])
def edit_memory(memory_id):
    """Edit an existing memory."""
    try:
        memory = db.load(memory_id)
        
        if request.method == 'POST':
            try:
                title = request.form.get('title', 'Untitled')
                category = request.form.get('category', 'Uncategorized')
                importance = float(request.form.get('importance', 0.5))
                
                # Update the memory node
                memory.set_attribute('title', title)
                memory.set_attribute('category', category)
                memory.set_attribute('importance', importance)
                
                # Parse embeddings from form if provided
                embedding_text = request.form.get('embeddings', '')
                if embedding_text:
                    try:
                        embeddings = json.loads(embedding_text)
                    except json.JSONDecodeError:
                        embeddings = [float(x) for x in embedding_text.split()]
                        
                    memory.set_embeddings(embeddings)
                
                # Save updated memory
                db.save(memory)
                
                flash(f"Memory '{title}' updated successfully!", 'success')
                return redirect(url_for('view_memory', memory_id=memory_id))
            except Exception as e:
                flash(f"Error updating memory: {str(e)}", 'error')
        
        # Prepare data for template
        memory_data = {
            'id': memory_id,
            'title': memory.get_attribute('title') or 'Untitled',
            'category': memory.get_attribute('category') or 'Uncategorized',
            'importance': memory.get_attribute('importance') or 0.0,
            'embeddings': get_embeddings_as_list(memory),
        }
                
        return render_template('edit.html', memory=memory_data)
    except Exception as e:
        flash(f"Error loading memory for editing: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/memory/<memory_id>/delete', methods=['POST'])
def delete_memory(memory_id):
    """Delete a memory."""
    try:
        memory = db.load(memory_id)
        title = memory.get_attribute('title') or 'Untitled'
        
        db.delete(memory_id)
        
        flash(f"Memory '{title}' deleted successfully!", 'success')
    except Exception as e:
        flash(f"Error deleting memory: {str(e)}", 'error')
        
    return redirect(url_for('index'))

@app.route('/search', methods=['GET', 'POST'])
def search():
    """Search for memories."""
    results = []
    
    if request.method == 'POST':
        try:
            # Get search parameters
            search_type = request.form.get('search_type', 'vector')
            
            if search_type == 'vector':
                # Vector similarity search
                query_text = request.form.get('query_text', '')
                use_text_embeddings = request.form.get('use_text_embeddings') == 'true'
                
                # Determine whether to use text or vector embeddings
                if use_text_embeddings and query_text and EMBEDDING_MODEL_AVAILABLE:
                    logger.info(f"Generating embeddings from query text: '{query_text}'")
                    query_vector = generate_embedding_from_query(query_text)
                    
                    if query_vector is None:
                        flash("Could not generate embeddings from query text. Using manual vector instead.", "warning")
                        # Fallback to manual vector input
                        embedding_text = request.form.get('query_vector', '')
                        try:
                            query_vector = json.loads(embedding_text)
                        except json.JSONDecodeError:
                            query_vector = [float(x) for x in embedding_text.split()]
                    else:
                        # Convert numpy array to list
                        query_vector = query_vector.tolist()
                else:
                    # Use the manual vector input
                    embedding_text = request.form.get('query_vector', '')
                    try:
                        query_vector = json.loads(embedding_text)
                    except json.JSONDecodeError:
                        query_vector = [float(x) for x in embedding_text.split()]
                
                threshold = float(request.form.get('threshold', 0.0))
                limit = int(request.form.get('limit', 10))
                
                # Perform search
                search_results = db.search_similar(query_vector, limit, threshold)
                
                # Load memory details
                for memory_id, similarity in search_results:
                    memory = db.load(memory_id)
                    memory_data = {
                        'id': memory_id,
                        'title': memory.get_attribute('title') or 'Untitled',
                        'category': memory.get_attribute('category') or 'Uncategorized',
                        'importance': memory.get_attribute('importance') or 0.0,
                        'content': memory.get_attribute('content') or '',
                        'similarity': similarity,
                    }
                    results.append(memory_data)
            else:
                # Attribute search (simple implementation)
                category = request.form.get('category', '')
                min_importance = float(request.form.get('min_importance', 0.0))
                
                # Get all memories and filter
                memory_ids = db.list_all()
                for memory_id in memory_ids:
                    memory = db.load(memory_id)
                    memory_category = memory.get_attribute('category') or 'Uncategorized'
                    memory_importance = memory.get_attribute('importance') or 0.0
                    
                    if (not category or memory_category == category) and memory_importance >= min_importance:
                        memory_data = {
                            'id': memory_id,
                            'title': memory.get_attribute('title') or 'Untitled',
                            'category': memory_category,
                            'importance': memory_importance,
                            'content': memory.get_attribute('content') or '',
                            'similarity': 'N/A',
                        }
                        results.append(memory_data)
                
        except Exception as e:
            flash(f"Error during search: {str(e)}", 'error')
    
    # Get categories for filter dropdown
    categories = set()
    try:
        memory_ids = db.list_all()
        for memory_id in memory_ids:
            memory = db.load(memory_id)
            category = memory.get_attribute('category')
            if category:
                categories.add(category)
    except Exception:
        pass
    
    return render_template('search.html', results=results, categories=list(categories), 
                          embedding_model_available=EMBEDDING_MODEL_AVAILABLE)

@app.errorhandler(404)
def page_not_found(e):
    flash('Page not found', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    flash('Server error', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)