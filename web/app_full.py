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
        # Store graph relationships as a dictionary:
        # {source_id: {target_id: {"type": relationship_type, "strength": strength}}}
        self.relationships = {}
    
    @classmethod
    def file_based(cls, path):
        return cls()
    
    @classmethod
    def in_memory(cls):
        return cls()
        
    def clear_all(self):
        """Delete all memories and relationships."""
        self.memories = {}
        self.relationships = {}
    
    def save(self, memory):
        self.memories[memory.id] = memory
        return memory.id
    
    def load(self, memory_id):
        if memory_id not in self.memories:
            raise ValueError(f"Memory with ID {memory_id} not found")
        return self.memories[memory_id]
    
    def delete(self, memory_id):
        if memory_id in self.memories:
            # Remove this memory
            del self.memories[memory_id]
            
            # Remove relationships where this memory is the source
            if memory_id in self.relationships:
                del self.relationships[memory_id]
            
            # Remove relationships where this memory is the target
            for source_id in self.relationships:
                if memory_id in self.relationships[source_id]:
                    del self.relationships[source_id][memory_id]
    
    def list_all(self):
        return list(self.memories.keys())
    
    def connect(self, source_id, target_id, relationship_type=None, strength=1.0):
        """Create a connection between two memories."""
        # Verify that both memories exist
        if source_id not in self.memories:
            raise ValueError(f"Source memory ID {source_id} not found")
        if target_id not in self.memories:
            raise ValueError(f"Target memory ID {target_id} not found")
        
        # Initialize the source dictionary if it doesn't exist
        if source_id not in self.relationships:
            self.relationships[source_id] = {}
        
        # Store the relationship
        self.relationships[source_id][target_id] = {
            "type": relationship_type,
            "strength": strength
        }
        
        return True
    
    def disconnect(self, source_id, target_id):
        """Remove a connection between two memories."""
        if source_id in self.relationships and target_id in self.relationships[source_id]:
            del self.relationships[source_id][target_id]
            return True
        return False
    
    def get_connections(self, memory_id, relationship_type=None):
        """Get all connections from a specific memory."""
        connections = []
        
        if memory_id in self.relationships:
            for target_id, rel_info in self.relationships[memory_id].items():
                if relationship_type is None or rel_info["type"] == relationship_type:
                    connections.append({
                        "target_id": target_id,
                        "type": rel_info["type"],
                        "strength": rel_info["strength"]
                    })
        
        return connections
    
    def get_connected_to(self, memory_id, relationship_type=None):
        """Get all memories that connect to this memory."""
        connections = []
        
        for source_id, targets in self.relationships.items():
            if memory_id in targets:
                rel_info = targets[memory_id]
                if relationship_type is None or rel_info["type"] == relationship_type:
                    connections.append({
                        "source_id": source_id,
                        "type": rel_info["type"],
                        "strength": rel_info["strength"]
                    })
        
        return connections
    
    def search_similar(self, query_vector, limit=10, threshold=0.0, connected_to=None, relationship_type=None):
        """Search for similar memories, optionally filtering by graph relationships."""
        # First, get all memories similar to the query vector
        import random
        from scipy.spatial.distance import cosine
        
        results = []
        for memory_id, memory in self.memories.items():
            # Skip if embedding dimensions don't match
            if len(memory._embeddings) != len(query_vector):
                continue
            
            # Calculate similarity score
            try:
                # Try to use cosine similarity if vectors have same dimensions
                similarity = 1 - cosine(memory._embeddings, query_vector)
            except:
                # Fallback to random similarity
                similarity = random.uniform(threshold, 1.0)
            
            # Only include if above threshold
            if similarity >= threshold:
                results.append((memory_id, similarity))
        
        # Filter by graph relationship if required
        if connected_to is not None:
            filtered_results = []
            
            # Get all memories connected to the specified memory
            connected_memories = set()
            if connected_to in self.relationships:
                for target_id, rel_info in self.relationships[connected_to].items():
                    if relationship_type is None or rel_info["type"] == relationship_type:
                        connected_memories.add(target_id)
            
            # Filter the results to include only connected memories
            for memory_id, similarity in results:
                if memory_id in connected_memories:
                    filtered_results.append((memory_id, similarity))
            
            results = filtered_results
        
        # Sort by similarity (highest first) and return top limit results
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]

try:
    # Try to import engramdb
    import engramdb_py
    from engramdb_py import MemoryNode, Database, RelationshipType
    logger.info("Successfully imported engramdb_py")
    # Test the MemoryNode to verify methods
    test_node = MemoryNode([0.1, 0.2])
    logger.info(f"Created test node with ID: {test_node.id}")
except ImportError as e:
    logger.warning(f"Using mock implementation: {e}")
    # Use mock implementations
    engramdb_py = None
    MemoryNode = MockMemoryNode
    Database = MockDatabase
    
    # Create a simple RelationshipType enum for mocks
    class RelationshipType:
        PREDECESSOR = "predecessor"
        SUCCESSOR = "successor"
        REFERENCE = "reference"
        ASSOCIATION = "association"

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

# Routes for graph/connection management
@app.route('/memory/<memory_id>/connections')
def view_connections(memory_id):
    """View all connections for a memory."""
    try:
        memory = db.load(memory_id)
        
        # Get outgoing connections (where this memory is the source)
        outgoing_connections = []
        try:
            # Get outgoing connections using the new implementation
            connections = db.get_connections(memory_id)
            for conn in connections:
                target = db.load(conn["target_id"])
                outgoing_connections.append({
                    'id': conn["target_id"],
                    'title': target.get_attribute('title') or 'Untitled',
                    'type': conn["type"].lower(),  # Lowercase for consistency
                    'strength': conn["strength"],
                    'direction': 'outgoing'
                })
        except Exception as e:
            logger.error(f"Error getting outgoing connections: {e}")
            # If there was an error with the new implementation, try the mock
            if hasattr(db, 'relationships') and memory_id in db.relationships:
                for target_id, rel_info in db.relationships[memory_id].items():
                    try:
                        target = db.load(target_id)
                        outgoing_connections.append({
                            'id': target_id,
                            'title': target.get_attribute('title') or 'Untitled',
                            'type': rel_info["type"].lower(),
                            'strength': rel_info["strength"],
                            'direction': 'outgoing'
                        })
                    except Exception as err:
                        logger.error(f"Error processing outgoing connection: {err}")
        
        # Get incoming connections (where this memory is the target)
        incoming_connections = []
        try:
            # Get incoming connections using the new implementation
            connections = db.get_connected_to(memory_id)
            for conn in connections:
                source = db.load(conn["source_id"])
                incoming_connections.append({
                    'id': conn["source_id"],
                    'title': source.get_attribute('title') or 'Untitled',
                    'type': conn["type"].lower(),
                    'strength': conn["strength"],
                    'direction': 'incoming'
                })
        except Exception as e:
            logger.error(f"Error getting incoming connections: {e}")
            # If there was an error with the new implementation, try using relationships
            if hasattr(db, 'relationships'):
                for source_id, targets in db.relationships.items():
                    if memory_id in targets:
                        try:
                            source = db.load(source_id)
                            rel_info = targets[memory_id]
                            incoming_connections.append({
                                'id': source_id,
                                'title': source.get_attribute('title') or 'Untitled',
                                'type': rel_info["type"].lower(),
                                'strength': rel_info["strength"],
                                'direction': 'incoming'
                            })
                        except Exception as err:
                            logger.error(f"Error processing incoming connection: {err}")
        
        # Get data for the current memory
        memory_data = {
            'id': memory_id,
            'title': memory.get_attribute('title') or 'Untitled',
            'category': memory.get_attribute('category') or 'Uncategorized',
            'importance': memory.get_attribute('importance') or 0.0,
        }
        
        # Get all other memories for the connection form
        all_memories = []
        for other_id in db.list_all():
            if other_id != memory_id:  # Exclude the current memory
                other = db.load(other_id)
                all_memories.append({
                    'id': other_id,
                    'title': other.get_attribute('title') or 'Untitled'
                })
        
        # Get relationship types
        try:
            # Check if we're using the actual Rust RelationshipType
            # or our mock Python implementation
            if hasattr(RelationshipType, 'Association'):
                # Rust RelationshipType enum
                relationship_types = [
                    {"id": "Association", "name": "Association"},
                    {"id": "Causation", "name": "Causation"},
                    {"id": "Sequence", "name": "Sequence"},
                    {"id": "PartOf", "name": "Part Of"},
                    {"id": "Contains", "name": "Contains"}
                ]
            else:
                # Mock RelationshipType class
                relationship_types = [
                    {"id": RelationshipType.ASSOCIATION, "name": "Association"},
                    {"id": RelationshipType.REFERENCE, "name": "Reference"},
                    {"id": RelationshipType.PREDECESSOR, "name": "Predecessor"},
                    {"id": RelationshipType.SUCCESSOR, "name": "Successor"}
                ]
        except Exception as e:
            logger.error(f"Error determining relationship types: {e}")
            # Fallback to basic relationships
            relationship_types = [
                {"id": "Association", "name": "Association"},
                {"id": "Reference", "name": "Reference"},
                {"id": "Predecessor", "name": "Predecessor"},
                {"id": "Successor", "name": "Successor"}
            ]
                
        return render_template('connections.html', 
                              memory=memory_data,
                              outgoing_connections=outgoing_connections,
                              incoming_connections=incoming_connections,
                              all_memories=all_memories,
                              relationship_types=relationship_types)
    except Exception as e:
        flash(f"Error loading connections: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/memory/<source_id>/connect', methods=['POST'])
def create_connection(source_id):
    """Create a new connection between memories."""
    try:
        target_id = request.form.get('target_id')
        relationship_type = request.form.get('relationship_type')
        strength = float(request.form.get('strength', 1.0))
        
        if not target_id:
            flash("Target memory is required", 'error')
            return redirect(url_for('view_connections', memory_id=source_id))
        
        # Check if we're using the real Database or mock
        if hasattr(db, 'connect'):
            # Using the mock database which has the connect method
            # Create the connection
            db.connect(source_id, target_id, relationship_type, strength)
            
            # If relationship is bidirectional, create reverse connection
            bidirectional = request.form.get('bidirectional') == 'true'
            if bidirectional:
                # Determine reverse relationship type
                reverse_type = relationship_type
                
                # Handle reverse relationships
                try:
                    # Check if we're using the mock implementation
                    if not hasattr(RelationshipType, 'Association'):
                        # Mock implementation with PREDECESSOR/SUCCESSOR
                        if relationship_type == RelationshipType.PREDECESSOR:
                            reverse_type = RelationshipType.SUCCESSOR
                        elif relationship_type == RelationshipType.SUCCESSOR:
                            reverse_type = RelationshipType.PREDECESSOR
                except Exception as e:
                    logger.warning(f"Error determining reverse relationship: {e}")
                
                db.connect(target_id, source_id, reverse_type, strength)
        else:
            # Real Database implementation
            logger.warning("Connection creation not yet implemented for real Database")
            flash("Connection creation not yet implemented in this version", "warning")
        
        flash("Connection created successfully", 'success')
    except Exception as e:
        flash(f"Error creating connection: {str(e)}", 'error')
    
    return redirect(url_for('view_connections', memory_id=source_id))

@app.route('/memory/<source_id>/disconnect/<target_id>', methods=['POST'])
def remove_connection(source_id, target_id):
    """Remove a connection between memories."""
    try:
        # Check if we're using the mock Database or real
        if hasattr(db, 'disconnect'):
            # Using the mock database
            db.disconnect(source_id, target_id)
            flash("Connection removed successfully", 'success')
        else:
            # Real Database implementation
            logger.warning("Connection removal not yet implemented for real Database")
            flash("Connection removal not yet implemented in this version", "warning")
    except Exception as e:
        flash(f"Error removing connection: {str(e)}", 'error')
    
    return redirect(url_for('view_connections', memory_id=source_id))

@app.route('/memory/graph')
def memory_graph():
    """View the entire memory graph."""
    try:
        # Get all memories for nodes
        nodes = []
        all_memory_ids = db.list_all()
        for memory_id in all_memory_ids:
            memory = db.load(memory_id)
            nodes.append({
                'id': memory_id,
                'title': memory.get_attribute('title') or 'Untitled',
                'category': memory.get_attribute('category') or 'Uncategorized',
                'importance': memory.get_attribute('importance') or 0.0,
            })
        
        # Get all connections for edges
        edges = []
        
        # For each memory, get its outgoing connections
        for memory_id in all_memory_ids:
            try:
                # Try using the new get_connections method
                connections = db.get_connections(memory_id)
                
                # Process each connection to create an edge
                for conn in connections:
                    edges.append({
                        'source': memory_id,
                        'target': conn["target_id"],
                        'type': conn["type"].lower(),  # Lowercase for consistency in visualization
                        'strength': conn["strength"]
                    })
            except Exception as e:
                logger.warning(f"Couldn't get connections for memory {memory_id}: {e}")
                # If any error occurs, try falling back to mock implementation
                if hasattr(db, 'relationships') and memory_id in db.relationships:
                    for target_id, rel_info in db.relationships[memory_id].items():
                        edges.append({
                            'source': memory_id,
                            'target': target_id,
                            'type': rel_info["type"].lower(),
                            'strength': rel_info["strength"]
                        })
        
        # If no edges were found, show a notification
        if not edges:
            flash("No connections found between memories. Create connections to see the graph.", "info")
            
        return render_template('graph.html', nodes=nodes, edges=edges)
    except Exception as e:
        flash(f"Error loading memory graph: {str(e)}", 'error')
        return redirect(url_for('index'))

@app.route('/search/graph', methods=['GET', 'POST'])
def graph_search():
    """Search for memories with graph relationship constraints."""
    results = []
    all_memories = []
    
    # Get all memories for dropdown
    try:
        for memory_id in db.list_all():
            memory = db.load(memory_id)
            all_memories.append({
                'id': memory_id,
                'title': memory.get_attribute('title') or 'Untitled'
            })
    except Exception:
        pass
        
    if request.method == 'POST':
        try:
            # Get search parameters
            query_text = request.form.get('query_text', '')
            connected_to = request.form.get('connected_to', '')
            relationship_type = request.form.get('relationship_type', '')
            
            # Set relationship_type to None if "Any" was selected
            if relationship_type == "any":
                relationship_type = None
                
            # Get query vector from text or manual input
            use_text_embeddings = request.form.get('use_text_embeddings') == 'true'
            if use_text_embeddings and query_text and EMBEDDING_MODEL_AVAILABLE:
                logger.info(f"Generating embeddings from query text: '{query_text}'")
                query_vector = generate_embedding_from_query(query_text)
                
                if query_vector is None:
                    flash("Could not generate embeddings from query text. Using manual vector instead.", "warning")
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
            
            # Check if the real Database supports graph constraints in search
            supports_graph_constraints = hasattr(db, 'relationships')
            
            # Perform search with graph constraints
            if connected_to and supports_graph_constraints:
                # Mock implementation supports graph constraints
                search_results = db.search_similar(
                    query_vector, 
                    limit=limit, 
                    threshold=threshold,
                    connected_to=connected_to,
                    relationship_type=relationship_type
                )
            else:
                # Real Database or search without constraints
                if connected_to and not supports_graph_constraints:
                    flash("Graph constraint search not available in this version", "warning")
                    
                search_results = db.search_similar(
                    query_vector, 
                    limit=limit, 
                    threshold=threshold
                )
            
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
                
        except Exception as e:
            flash(f"Error during graph search: {str(e)}", 'error')
    
    # Get relationship types for dropdown
    try:
        # Check if we're using the actual Rust RelationshipType
        # or our mock Python implementation
        if hasattr(RelationshipType, 'Association'):
            # Rust RelationshipType enum
            relationship_types = [
                {"id": "any", "name": "Any Relationship"},
                {"id": "Association", "name": "Association"},
                {"id": "Causation", "name": "Causation"},
                {"id": "Sequence", "name": "Sequence"},
                {"id": "PartOf", "name": "Part Of"},
                {"id": "Contains", "name": "Contains"}
            ]
        else:
            # Mock RelationshipType class
            relationship_types = [
                {"id": "any", "name": "Any Relationship"},
                {"id": RelationshipType.ASSOCIATION, "name": "Association"},
                {"id": RelationshipType.REFERENCE, "name": "Reference"},
                {"id": RelationshipType.PREDECESSOR, "name": "Predecessor"},
                {"id": RelationshipType.SUCCESSOR, "name": "Successor"}
            ]
    except Exception as e:
        logger.error(f"Error determining relationship types: {e}")
        # Fallback to basic relationships
        relationship_types = [
            {"id": "any", "name": "Any Relationship"},
            {"id": "Association", "name": "Association"},
            {"id": "Reference", "name": "Reference"},
            {"id": "Predecessor", "name": "Predecessor"},
            {"id": "Successor", "name": "Successor"}
        ]
    
    return render_template(
        'graph_search.html', 
        results=results, 
        all_memories=all_memories, 
        relationship_types=relationship_types,
        embedding_model_available=EMBEDDING_MODEL_AVAILABLE
    )

@app.errorhandler(404)
def page_not_found(e):
    flash('Page not found', 'error')
    return redirect(url_for('index'))

@app.route('/delete_all_memories', methods=['POST'])
def delete_all_memories():
    """Delete all memories and connections."""
    global db
    try:
        # First, try the standard approach
        memory_ids = db.list_all()
        success_count = 0
        error_count = 0
        
        for memory_id in memory_ids:
            try:
                db.delete(memory_id)
                success_count += 1
            except Exception as e:
                logger.error(f"Failed to delete memory {memory_id}: {e}")
                error_count += 1
        
        # If we encountered errors or if there are still files, do a hard reset
        if error_count > 0:
            logger.info("Performing manual database reset")
            
            # Create a new database instance
            try:
                # First, let's try to manually delete the files if they exist
                import shutil
                import os
                
                # Delete files in the database directory
                if os.path.exists(DB_PATH):
                    # Remove all subdirectories and files
                    for root, dirs, files in os.walk(DB_PATH):
                        for f in files:
                            try:
                                os.unlink(os.path.join(root, f))
                                logger.info(f"Deleted file: {os.path.join(root, f)}")
                            except Exception as e:
                                logger.error(f"Failed to delete file {f}: {e}")
                    
                    # Recreate the directory structure
                    os.makedirs(DB_PATH, exist_ok=True)
                
                # Reinitialize the database
                db = Database.file_based(DB_PATH)
                logger.info("Database reinitialized successfully")
                flash("Database has been completely reset", 'success')
            except Exception as e:
                logger.error(f"Error during manual database reset: {e}")
                flash(f"Error resetting database: {str(e)}", 'error')
        else:
            flash(f"All {success_count} memories have been deleted", 'success')
    except Exception as e:
        flash(f"Error deleting memories: {str(e)}", 'error')
    
    return redirect(url_for('index'))

@app.route('/load_example_dataset', methods=['POST'])
def load_example_dataset():
    """
    Load sample AI assistant memory dataset.
    
    This endpoint uses the Rust-based implementation in examples/sample_dataset.rs to create
    a realistic AI assistant memory graph with:
    - User profile information
    - User preferences
    - Past conversations
    - Technical knowledge
    - Resource recommendations
    - Current project context
    - Meaningful connections between all nodes
    """
    try:
        # Debug output to check the module
        import inspect
        logger.info(f"engramdb_py type: {type(engramdb_py)}")
        logger.info(f"engramdb_py dir: {dir(engramdb_py) if engramdb_py else 'None'}")
        
        # First clear any existing data
        try:
            logger.info("Clearing existing database contents...")
            db.clear_all()
        except Exception as e:
            logger.warning(f"Error clearing database with clear_all(): {e}")
            # Fallback to manual deletion
            memory_ids = db.list_all()
            logger.info(f"Falling back to manual deletion of {len(memory_ids)} nodes")
            for memory_id in memory_ids:
                try:
                    db.delete(memory_id)
                except Exception as err:
                    logger.warning(f"Could not delete node {memory_id}: {err}")
        
        # Use the Rust-based implementation from examples/sample_dataset.rs
        logger.info("Loading sample dataset using Rust implementation from examples/sample_dataset.rs")
        node_ids = engramdb_py.load_sample_dataset(db)
        logger.info(f"Successfully created {len(node_ids)} memory nodes")
        
        flash("AI assistant memory dataset loaded successfully! The dataset includes user profile, preferences, " + 
              "conversations, knowledge, and project information with meaningful connections.", "success")
    except Exception as e:
        logger.error(f"Error loading example dataset: {e}", exc_info=True)
        flash(f"Error loading example dataset: {e}", "error")
        
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    flash('Server error', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)