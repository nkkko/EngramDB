import os
import sys
import json
import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.exceptions import NotFound
import uuid
import logging

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
    import numpy as np
    try:
        # Get and convert to standard Python list
        embeddings = memory_node.get_embeddings()
        return embeddings.tolist()
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

try:
    # Import engramdb
    from engramdb_py import MemoryNode, Database, RelationshipType
    logger.info("Successfully imported engramdb_py")
    # Test the MemoryNode to verify methods
    test_node = MemoryNode([0.1, 0.2])
    logger.info(f"Created test node with ID: {test_node.id}")  # Access as property, not method
    # Note: get_embeddings requires Python GIL parameter, we'll handle differently in routes
    logger.info(f"Available methods: {dir(test_node)}")
except ImportError as e:
    logger.error(f"Error importing engramdb_py: {e}")
    sys.exit(1)
except Exception as e:
    logger.error(f"Unexpected error testing EngramDB: {e}")
    sys.exit(1)

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
            
            # Save to database
            memory_id = db.save(memory)
            
            flash(f"Memory '{title}' created successfully!", 'success')
            return redirect(url_for('view_memory', memory_id=memory_id))
        except Exception as e:
            flash(f"Error creating memory: {str(e)}", 'error')
            
    return render_template('create.html')

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
    
    return render_template('search.html', results=results, categories=list(categories))

@app.errorhandler(404)
def page_not_found(e):
    flash('Page not found', 'error')
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    flash('Server error', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)