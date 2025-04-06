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
    """Load realistic example dataset with AI Coding Agent bug fixing workflow including multiple attempts."""
    try:
        # First, clear any existing memories
        try:
            db.clear_all()
        except Exception as e:
            logger.warning(f"Error clearing database: {e}")
            # Fallback to manual deletion
            memory_ids = db.list_all()
            for memory_id in memory_ids:
                try:
                    db.delete(memory_id)
                except Exception:
                    pass
        
        # Create memories for AI Coding Agent bug fixing workflow
        # This will be a complex, realistic bug fixing process with:
        # - Initial analysis
        # - Failed attempt 1 (wrong solution path)
        # - Failed attempt 2 (partial solution)
        # - Deeper investigation
        # - Final successful solution
        
        #--------------------- INITIAL BUG REPORT AND ANALYSIS ---------------------#
        
        # Memory 1: Bug Report
        if EMBEDDING_MODEL_AVAILABLE:
            bug_report = MemoryNode(generate_embedding_for_memory(
                "NullPointerException in UserService.java at line 45 when calling getUserDetails() in production environment",
                "bug_report"
            ).tolist())
        else:
            bug_report = MemoryNode([0.82, 0.41, 0.33, 0.25, 0.1])
            
        bug_report.set_attribute('title', 'NullPointerException in UserService')
        bug_report.set_attribute('category', 'bug_report')
        bug_report.set_attribute('importance', 0.9)
        bug_report.set_attribute('content', "NullPointerException in UserService.java at line 45 when calling getUserDetails() in production environment. Error occurs inconsistently, making it difficult to reproduce reliably.")
        bug_report.set_attribute('memory_type', "bug_report")
        bug_report.set_attribute('module', "user-management")
        bug_report.set_attribute('timestamp', "2025-04-01T09:15:00")
        bug_report_id = db.save(bug_report)
        
        # Memory 2: Initial Codebase Exploration
        if EMBEDDING_MODEL_AVAILABLE:
            codebase_memory = MemoryNode(generate_embedding_for_memory(
                "Traversed UserService.java, identified dependencies on UserRepository.java and AuthorizationService.java", 
                "codebase_structure"
            ).tolist())
        else:
            codebase_memory = MemoryNode([0.74, 0.52, 0.28, 0.31, 0.15])
            
        codebase_memory.set_attribute('title', 'UserService Class Structure')
        codebase_memory.set_attribute('category', 'codebase_structure')
        codebase_memory.set_attribute('importance', 0.7)
        codebase_memory.set_attribute('content', "Traversed UserService.java and found dependencies on UserRepository.java and AuthorizationService.java. Line 45 contains: return userRepository.findUser(id).getDetails();")
        codebase_memory.set_attribute('memory_type', "codebase_structure")
        codebase_memory.set_attribute('module', "user-management")
        codebase_memory.set_attribute('timestamp', "2025-04-01T09:20:00")
        codebase_id = db.save(codebase_memory)
        
        # Memory 3: Initial Bug Analysis
        if EMBEDDING_MODEL_AVAILABLE:
            analysis_memory = MemoryNode(generate_embedding_for_memory(
                "Possible reasons for NullPointerException: 1) userRepository is null, 2) findUser() returned null, 3) user.getDetails() called on null object",
                "analysis_result"
            ).tolist())
        else:
            analysis_memory = MemoryNode([0.85, 0.44, 0.30, 0.22, 0.18])
            
        analysis_memory.set_attribute('title', 'Initial Root Cause Analysis')
        analysis_memory.set_attribute('category', 'analysis_result')
        analysis_memory.set_attribute('importance', 0.85)
        analysis_memory.set_attribute('content', "Three possible causes for NullPointerException:\n1. userRepository field is null\n2. findUser() returned null\n3. getDetails() called on null user object\n\nSince error occurs inconsistently, most likely scenario is that findUser() sometimes returns null.")
        analysis_memory.set_attribute('memory_type', "analysis_result")
        analysis_memory.set_attribute('module', "user-management")
        analysis_memory.set_attribute('confidence', 0.65)
        analysis_memory.set_attribute('timestamp', "2025-04-01T09:30:00")
        initial_analysis_id = db.save(analysis_memory)
        
        #--------------------- FAILED ATTEMPT 1 (WRONG SOLUTION PATH) ---------------------#
        
        # Memory 4: First Solution Attempt (incorrect hypothesis)
        if EMBEDDING_MODEL_AVAILABLE:
            solution1_memory = MemoryNode(generate_embedding_for_memory(
                "Add null check before calling getDetails() to prevent NullPointerException when findUser() returns null",
                "proposed_fix"
            ).tolist())
        else: 
            solution1_memory = MemoryNode([0.79, 0.51, 0.34, 0.28, 0.22])
            
        solution1_memory.set_attribute('title', 'Null Check Fix (Attempt 1)')
        solution1_memory.set_attribute('category', 'proposed_fix')
        solution1_memory.set_attribute('importance', 0.8)
        solution1_memory.set_attribute('content', "Add null check before calling getDetails() to prevent NullPointerException when findUser() returns null:\n\nUser user = userRepository.findUser(id);\nreturn user != null ? user.getDetails() : null;")
        solution1_memory.set_attribute('memory_type', "proposed_fix")
        solution1_memory.set_attribute('module', "user-management")
        solution1_memory.set_attribute('estimated_success', 0.7)
        solution1_memory.set_attribute('timestamp', "2025-04-01T09:45:00")
        solution1_id = db.save(solution1_memory)
        
        # Memory 5: First Implementation
        if EMBEDDING_MODEL_AVAILABLE:
            implementation1_memory = MemoryNode(generate_embedding_for_memory(
                "User user = userRepository.findUser(id); return user != null ? user.getDetails() : null;",
                "code_snippet"
            ).tolist())
        else: 
            implementation1_memory = MemoryNode([0.76, 0.48, 0.37, 0.31, 0.25])
            
        implementation1_memory.set_attribute('title', 'Null Check Implementation')
        implementation1_memory.set_attribute('category', 'code_snippet')
        implementation1_memory.set_attribute('importance', 0.8)
        implementation1_memory.set_attribute('content', """// Modified getUserDetails() method
public UserDetails getUserDetails(String id) {
    User user = userRepository.findUser(id);
    return user != null ? user.getDetails() : null;
}""")
        implementation1_memory.set_attribute('memory_type', "code_snippet")
        implementation1_memory.set_attribute('module', "user-management")
        implementation1_memory.set_attribute('file', "UserService.java")
        implementation1_memory.set_attribute('timestamp', "2025-04-01T10:00:00")
        implementation1_id = db.save(implementation1_memory)
        
        # Memory 6: First Verification (failed)
        if EMBEDDING_MODEL_AVAILABLE:
            verification1_memory = MemoryNode(generate_embedding_for_memory(
                "Fix partially successful but NullPointerException still occurs in some edge cases. Error now points to line 32 in same file.",
                "testing_outcome"
            ).tolist())
        else: 
            verification1_memory = MemoryNode([0.72, 0.45, 0.33, 0.35, 0.28])
            
        verification1_memory.set_attribute('title', 'Failed Verification (Attempt 1)')
        verification1_memory.set_attribute('category', 'testing_outcome')
        verification1_memory.set_attribute('importance', 0.8)
        verification1_memory.set_attribute('content', "Fix partially successful but NullPointerException still occurs in some edge cases. Error now points to line 32 in same file: userRepository.updateLastAccess(id). The first hypothesis was incomplete - userRepository itself may be null.")
        verification1_memory.set_attribute('memory_type', "testing_outcome")
        verification1_memory.set_attribute('module', "user-management")
        verification1_memory.set_attribute('success', False)
        verification1_memory.set_attribute('timestamp', "2025-04-01T10:30:00")
        verification1_id = db.save(verification1_memory)
        
        #--------------------- FAILED ATTEMPT 2 (MISSING DEEPER ISSUE) ---------------------#
        
        # Memory 7: Second Analysis (deeper but still incomplete)
        if EMBEDDING_MODEL_AVAILABLE:
            analysis2_memory = MemoryNode(generate_embedding_for_memory(
                "Main issue: userRepository itself is null. Need to check how UserRepository is initialized. May need constructor injection or null checks at all usages.",
                "analysis_result"
            ).tolist())
        else:
            analysis2_memory = MemoryNode([0.86, 0.41, 0.32, 0.24, 0.19])
            
        analysis2_memory.set_attribute('title', 'Revised Root Cause Analysis')
        analysis2_memory.set_attribute('category', 'analysis_result')
        analysis2_memory.set_attribute('importance', 0.9)
        analysis2_memory.set_attribute('content', "Main issue appears to be that userRepository itself is null. Need to investigate how UserRepository is initialized in UserService class. This is likely caused by incorrect dependency injection. Error occurs inconsistently because it may depend on initialization order.")
        analysis2_memory.set_attribute('memory_type', "analysis_result")
        analysis2_memory.set_attribute('module', "user-management")
        analysis2_memory.set_attribute('confidence', 0.85)
        analysis2_memory.set_attribute('timestamp', "2025-04-01T11:00:00")
        analysis2_id = db.save(analysis2_memory)
        
        # Memory 8: Deeper Code Exploration
        if EMBEDDING_MODEL_AVAILABLE:
            codeExplore2_memory = MemoryNode(generate_embedding_for_memory(
                "UserService.java lacks proper constructor injection. It uses field injection without initialization. UserRepository only initialized by setter method setUserRepository() which may not be called consistently.",
                "codebase_structure"
            ).tolist())
        else:
            codeExplore2_memory = MemoryNode([0.73, 0.50, 0.29, 0.30, 0.16])
            
        codeExplore2_memory.set_attribute('title', 'Dependency Injection Pattern Analysis')
        codeExplore2_memory.set_attribute('category', 'codebase_structure')
        codeExplore2_memory.set_attribute('importance', 0.85)
        codeExplore2_memory.set_attribute('content', "UserService.java uses field injection without proper initialization. It has a @Autowired annotation on userRepository field but no constructor injection. May indicate configuration issue. Found setter method setUserRepository() that gets called by Spring in some contexts but not others.")
        codeExplore2_memory.set_attribute('memory_type', "codebase_structure")
        codeExplore2_memory.set_attribute('module', "user-management")
        codeExplore2_memory.set_attribute('timestamp', "2025-04-01T11:15:00")
        codeExplore2_id = db.save(codeExplore2_memory)
        
        # Memory 9: Second Solution Attempt (closer but incomplete)
        if EMBEDDING_MODEL_AVAILABLE:
            solution2_memory = MemoryNode(generate_embedding_for_memory(
                "Add null checks for userRepository before every use, falling back to empty results when null",
                "proposed_fix"
            ).tolist())
        else: 
            solution2_memory = MemoryNode([0.78, 0.53, 0.32, 0.29, 0.23])
            
        solution2_memory.set_attribute('title', 'Defensive Null Checks (Attempt 2)')
        solution2_memory.set_attribute('category', 'proposed_fix')
        solution2_memory.set_attribute('importance', 0.8)
        solution2_memory.set_attribute('content', "Add comprehensive null checks for userRepository before every use, falling back to empty results when null. This should prevent the NullPointerException but doesn't address the root cause of userRepository not being properly initialized.")
        solution2_memory.set_attribute('memory_type', "proposed_fix")
        solution2_memory.set_attribute('module', "user-management")
        solution2_memory.set_attribute('estimated_success', 0.75)
        solution2_memory.set_attribute('timestamp', "2025-04-01T11:30:00")
        solution2_id = db.save(solution2_memory)
        
        # Memory 10: Second Implementation
        if EMBEDDING_MODEL_AVAILABLE:
            implementation2_memory = MemoryNode(generate_embedding_for_memory(
                "Added null checks for userRepository before all method calls to prevent NullPointerException",
                "code_snippet"
            ).tolist())
        else: 
            implementation2_memory = MemoryNode([0.75, 0.49, 0.38, 0.33, 0.24])
            
        implementation2_memory.set_attribute('title', 'Defensive Null Checks Implementation')
        implementation2_memory.set_attribute('category', 'code_snippet')
        implementation2_memory.set_attribute('importance', 0.8)
        implementation2_memory.set_attribute('content', """// Modified getUserDetails() with comprehensive null check
public UserDetails getUserDetails(String id) {
    if (userRepository == null) {
        logger.warn("UserRepository is null when calling getUserDetails");
        return null;
    }
    User user = userRepository.findUser(id);
    return user != null ? user.getDetails() : null;
}

// Also modified updateLastAccess() with null check
public void updateLastAccess(String id) {
    if (userRepository == null) {
        logger.warn("UserRepository is null when calling updateLastAccess");
        return;
    }
    userRepository.updateLastAccess(id);
}""")
        implementation2_memory.set_attribute('memory_type', "code_snippet")
        implementation2_memory.set_attribute('module', "user-management")
        implementation2_memory.set_attribute('file', "UserService.java")
        implementation2_memory.set_attribute('timestamp', "2025-04-01T11:45:00")
        implementation2_id = db.save(implementation2_memory)
        
        # Memory 11: Second Verification (better but still issues)
        if EMBEDDING_MODEL_AVAILABLE:
            verification2_memory = MemoryNode(generate_embedding_for_memory(
                "NullPointerExceptions fixed but application misbehaves due to returning null instead of finding users. Need proper initialization instead of defensive coding.",
                "testing_outcome"
            ).tolist())
        else: 
            verification2_memory = MemoryNode([0.71, 0.46, 0.34, 0.36, 0.29])
            
        verification2_memory.set_attribute('title', 'Partial Success Verification (Attempt 2)')
        verification2_memory.set_attribute('category', 'testing_outcome')
        verification2_memory.set_attribute('importance', 0.85)
        verification2_memory.set_attribute('content', "NullPointerExceptions no longer occur, but application misbehaves due to silently returning null instead of properly finding users. Logs show frequent 'UserRepository is null' warnings. Fix treats symptoms but doesn't address root cause: proper dependency initialization.")
        verification2_memory.set_attribute('memory_type', "testing_outcome")
        verification2_memory.set_attribute('module', "user-management")
        verification2_memory.set_attribute('success', False)
        verification2_memory.set_attribute('timestamp', "2025-04-01T13:00:00")
        verification2_id = db.save(verification2_memory)
        
        #--------------------- DEEPER INVESTIGATION ---------------------#
        
        # Memory 12: Dependency Injection Investigation
        if EMBEDDING_MODEL_AVAILABLE:
            investigation_memory = MemoryNode(generate_embedding_for_memory(
                "Investigated Spring dependency injection patterns. Field injection with @Autowired is unreliable compared to constructor injection. Best practice is to use constructor injection to ensure all dependencies initialized at creation.",
                "analysis_result"
            ).tolist())
        else:
            investigation_memory = MemoryNode([0.88, 0.43, 0.29, 0.25, 0.17])
            
        investigation_memory.set_attribute('title', 'Dependency Injection Investigation')
        investigation_memory.set_attribute('category', 'analysis_result')
        investigation_memory.set_attribute('importance', 0.95)
        investigation_memory.set_attribute('content', "Researched Spring dependency injection patterns. Field-level @Autowired annotation is less reliable than constructor injection. Spring documentation recommends constructor injection to ensure all required dependencies are initialized at object creation time. Field injection can lead to partially initialized objects if circular dependencies exist.")
        investigation_memory.set_attribute('memory_type', "analysis_result")
        investigation_memory.set_attribute('module', "user-management")
        investigation_memory.set_attribute('confidence', 0.95)
        investigation_memory.set_attribute('timestamp', "2025-04-01T14:30:00")
        investigation_id = db.save(investigation_memory)
        
        # Memory 13: Application Context Investigation
        if EMBEDDING_MODEL_AVAILABLE:
            context_memory = MemoryNode(generate_embedding_for_memory(
                "Traced UserService lifecycle in application context. UserService sometimes created by manual instantiation without dependency injection. Found places where 'new UserService()' is called directly.",
                "analysis_result"
            ).tolist())
        else:
            context_memory = MemoryNode([0.84, 0.45, 0.32, 0.20, 0.16])
            
        context_memory.set_attribute('title', 'Application Context Analysis')
        context_memory.set_attribute('category', 'analysis_result')
        context_memory.set_attribute('importance', 0.95)
        context_memory.set_attribute('content', "Traced UserService lifecycle in application context. Found critical issue: UserService is sometimes created by manual instantiation without dependency injection. Found code in AdminController.java where 'new UserService()' is called directly, bypassing Spring's dependency injection. This explains the inconsistent failures.")
        context_memory.set_attribute('memory_type', "analysis_result")
        context_memory.set_attribute('module', "user-management")
        context_memory.set_attribute('confidence', 0.98)
        context_memory.set_attribute('timestamp', "2025-04-01T15:30:00")
        context_id = db.save(context_memory)
        
        #--------------------- FINAL SOLUTION ---------------------#
        
        # Memory 14: Comprehensive Solution
        if EMBEDDING_MODEL_AVAILABLE:
            final_solution_memory = MemoryNode(generate_embedding_for_memory(
                "Two-part solution: 1) Refactor UserService to use constructor injection, 2) Remove direct instantiation in AdminController and use proper dependency injection",
                "proposed_fix"
            ).tolist())
        else: 
            final_solution_memory = MemoryNode([0.89, 0.50, 0.33, 0.26, 0.21])
            
        final_solution_memory.set_attribute('title', 'Complete Dependency Injection Fix')
        final_solution_memory.set_attribute('category', 'proposed_fix')
        final_solution_memory.set_attribute('importance', 0.95)
        final_solution_memory.set_attribute('content', "Two-part comprehensive solution:\n1) Refactor UserService to use constructor injection instead of field injection\n2) Fix AdminController to properly inject UserService instead of creating with 'new'\n\nThis addresses the root cause by ensuring userRepository is always initialized before UserService methods are called.")
        final_solution_memory.set_attribute('memory_type', "proposed_fix")
        final_solution_memory.set_attribute('module', "user-management")
        final_solution_memory.set_attribute('estimated_success', 0.98)
        final_solution_memory.set_attribute('timestamp', "2025-04-01T16:00:00")
        final_solution_id = db.save(final_solution_memory)
        
        # Memory 15: UserService Implementation
        if EMBEDDING_MODEL_AVAILABLE:
            final_impl1_memory = MemoryNode(generate_embedding_for_memory(
                "Refactored UserService to use constructor injection for all dependencies including UserRepository and AuthorizationService",
                "code_snippet"
            ).tolist())
        else: 
            final_impl1_memory = MemoryNode([0.77, 0.52, 0.36, 0.30, 0.23])
            
        final_impl1_memory.set_attribute('title', 'UserService Constructor Injection')
        final_impl1_memory.set_attribute('category', 'code_snippet')
        final_impl1_memory.set_attribute('importance', 0.9)
        final_impl1_memory.set_attribute('content', """// Refactored UserService with constructor injection
@Service
public class UserService {
    private final UserRepository userRepository;
    private final AuthorizationService authService;
    private final Logger logger = LoggerFactory.getLogger(UserService.class);
    
    @Autowired
    public UserService(UserRepository userRepository, AuthorizationService authService) {
        this.userRepository = userRepository;
        this.authService = authService;
    }
    
    public UserDetails getUserDetails(String id) {
        User user = userRepository.findUser(id);
        return user != null ? user.getDetails() : null;
    }
    
    public void updateLastAccess(String id) {
        userRepository.updateLastAccess(id);
    }
}""")
        final_impl1_memory.set_attribute('memory_type', "code_snippet")
        final_impl1_memory.set_attribute('module', "user-management")
        final_impl1_memory.set_attribute('file', "UserService.java")
        final_impl1_memory.set_attribute('timestamp', "2025-04-01T16:15:00")
        final_impl1_id = db.save(final_impl1_memory)
        
        # Memory 16: AdminController Implementation
        if EMBEDDING_MODEL_AVAILABLE:
            final_impl2_memory = MemoryNode(generate_embedding_for_memory(
                "Fixed AdminController to use dependency injection for UserService instead of direct instantiation",
                "code_snippet"
            ).tolist())
        else: 
            final_impl2_memory = MemoryNode([0.75, 0.51, 0.39, 0.32, 0.24])
            
        final_impl2_memory.set_attribute('title', 'AdminController Dependency Fix')
        final_impl2_memory.set_attribute('category', 'code_snippet')
        final_impl2_memory.set_attribute('importance', 0.9)
        final_impl2_memory.set_attribute('content', """// Fixed AdminController to properly inject UserService
@Controller
@RequestMapping("/admin")
public class AdminController {
    private final UserService userService;
    
    @Autowired
    public AdminController(UserService userService) {
        this.userService = userService;
    }
    
    @GetMapping("/user/{id}")
    public String getUserDetails(@PathVariable String id, Model model) {
        // No longer using "new UserService()" here
        UserDetails details = userService.getUserDetails(id);
        model.addAttribute("userDetails", details);
        return "user/details";
    }
}""")
        final_impl2_memory.set_attribute('memory_type', "code_snippet")
        final_impl2_memory.set_attribute('module', "user-management")
        final_impl2_memory.set_attribute('file', "AdminController.java")
        final_impl2_memory.set_attribute('timestamp', "2025-04-01T16:30:00")
        final_impl2_id = db.save(final_impl2_memory)
        
        # Memory 17: Final Verification
        if EMBEDDING_MODEL_AVAILABLE:
            final_verification_memory = MemoryNode(generate_embedding_for_memory(
                "All tests pass. No NullPointerExceptions in production or testing. Dependency injection properly working in all contexts.",
                "testing_outcome"
            ).tolist())
        else: 
            final_verification_memory = MemoryNode([0.73, 0.47, 0.32, 0.34, 0.27])
            
        final_verification_memory.set_attribute('title', 'Successful Final Verification')
        final_verification_memory.set_attribute('category', 'testing_outcome')
        final_verification_memory.set_attribute('importance', 0.95)
        final_verification_memory.set_attribute('content', "All tests pass. No NullPointerExceptions in comprehensive test suite or production monitoring. Dependency injection properly working in all contexts. Added integration test to specifically verify object creation paths. Fix addresses root cause by ensuring proper initialization in all contexts.")
        final_verification_memory.set_attribute('memory_type', "testing_outcome")
        final_verification_memory.set_attribute('module', "user-management")
        final_verification_memory.set_attribute('success', True)
        final_verification_memory.set_attribute('timestamp', "2025-04-01T17:30:00")
        final_verification_id = db.save(final_verification_memory)
        
        # Memory 18: Learning Reflection
        if EMBEDDING_MODEL_AVAILABLE:
            reflection_memory = MemoryNode(generate_embedding_for_memory(
                "Analysis of bug fixing process. Key learning: Field injection can cause subtle initialization problems. Always use constructor injection for required dependencies. Integration tests should verify object creation paths.",
                "reflection"
            ).tolist())
        else: 
            reflection_memory = MemoryNode([0.81, 0.55, 0.35, 0.29, 0.20])
            
        reflection_memory.set_attribute('title', 'Dependency Injection Best Practices')
        reflection_memory.set_attribute('category', 'reflection')
        reflection_memory.set_attribute('importance', 0.9)
        reflection_memory.set_attribute('content', """Key learnings from this bug fix:
1. Always use constructor injection over field injection for required dependencies
2. Never directly instantiate objects that require dependency injection
3. Create integration tests that verify all object creation paths
4. Null checks are band-aids - fix the root cause of improper initialization
5. Log when unexpected null values are encountered to help debugging

This pattern of error is common in Spring applications where developers mix dependency injection with direct instantiation.""")
        reflection_memory.set_attribute('memory_type', "reflection")
        reflection_memory.set_attribute('module', "user-management")
        reflection_memory.set_attribute('timestamp', "2025-04-01T18:00:00")
        reflection_id = db.save(reflection_memory)
        
        #--------------------- CONNECTIONS BETWEEN MEMORIES ---------------------#
        
        # Initial workflow - using CAUSATION for "led to" relationships
        db.connect(bug_report_id, codebase_id, "CAUSATION", 0.95)
        db.connect(codebase_id, initial_analysis_id, "CAUSATION", 0.9)
        
        # First attempt (failure path)
        db.connect(initial_analysis_id, solution1_id, "CAUSATION", 0.8)
        db.connect(solution1_id, implementation1_id, "REFERENCE", 0.9)  # Implementation references solution
        db.connect(implementation1_id, verification1_id, "CAUSATION", 0.9)  # Implementation causes verification results
        db.connect(verification1_id, analysis2_id, "CAUSATION", 0.9)  # Failure caused revised analysis
        
        # Second attempt (partial success but failure)
        db.connect(analysis2_id, codeExplore2_id, "CAUSATION", 0.9)  # Analysis caused deeper exploration
        db.connect(codeExplore2_id, solution2_id, "CAUSATION", 0.9)
        db.connect(solution2_id, implementation2_id, "REFERENCE", 0.9)  # Implementation references solution
        db.connect(implementation2_id, verification2_id, "CAUSATION", 0.9)
        db.connect(verification2_id, investigation_id, "CAUSATION", 0.9)  # Partial failure caused deeper investigation
        
        # Deep investigation
        db.connect(investigation_id, context_id, "CAUSATION", 0.95)
        
        # Final solution - using various relationship types appropriately
        db.connect(context_id, final_solution_id, "CAUSATION", 0.95)
        db.connect(final_solution_id, final_impl1_id, "REFERENCE", 0.9)  # Implementation references solution
        db.connect(final_solution_id, final_impl2_id, "REFERENCE", 0.9)  # Implementation references solution
        db.connect(final_impl1_id, final_verification_id, "CAUSATION", 0.9)  # Implementation caused verification results
        db.connect(final_impl2_id, final_verification_id, "CAUSATION", 0.9)  # Implementation caused verification results
        db.connect(final_verification_id, reflection_id, "CAUSATION", 0.9)  # Verification led to reflection
        
        # Direct connections showing relationship to original bug report - using REFERENCE
        db.connect(bug_report_id, verification1_id, "REFERENCE", 0.4)  # Bug report referenced by verification
        db.connect(bug_report_id, verification2_id, "REFERENCE", 0.7)  # Bug report referenced by verification
        db.connect(bug_report_id, final_verification_id, "REFERENCE", 0.98)  # Bug report referenced by verification
        
        # Cross-connections between analysis steps - using SEQUENCE for step improvements
        db.connect(initial_analysis_id, analysis2_id, "SEQUENCE", 0.7)  # Analysis steps in sequence
        db.connect(analysis2_id, investigation_id, "SEQUENCE", 0.8)  # Analysis steps in sequence
        db.connect(investigation_id, final_solution_id, "CAUSATION", 0.9)  # Investigation informed solution
        
        # Alternative path connections - using ASSOCIATION for alternative approaches
        db.connect(solution1_id, solution2_id, "ASSOCIATION", 0.6)  # Associated alternative solutions
        db.connect(solution2_id, final_solution_id, "ASSOCIATION", 0.8)  # Associated alternative solutions
        
        # Implementation relationships - using SEQUENCE for progressive improvements
        db.connect(implementation1_id, implementation2_id, "SEQUENCE", 0.7)  # Implementation sequence/progression
        db.connect(implementation2_id, final_impl1_id, "SEQUENCE", 0.9)  # Implementation sequence/progression
        
        # Solution contains implementations - using CONTAINS relationship
        db.connect(final_solution_id, final_impl1_id, "CONTAINS", 0.9)  # Solution contains implementation 1
        db.connect(final_solution_id, final_impl2_id, "CONTAINS", 0.9)  # Solution contains implementation 2
        
        # Memory structure relationships - using PART_OF
        db.connect(implementation1_id, solution1_id, "PART_OF", 0.9)  # Implementation is part of solution
        db.connect(implementation2_id, solution2_id, "PART_OF", 0.9)  # Implementation is part of solution
        
        # Temporal relationships - using PREDECESSOR/SUCCESSOR
        db.connect(verification1_id, verification2_id, "PREDECESSOR", 0.8)  # Verification 1 preceded verification 2
        db.connect(verification2_id, final_verification_id, "PREDECESSOR", 0.8)  # Verification 2 preceded final verification
        
        # Lessons learned connection - using REFERENCE
        db.connect(reflection_id, bug_report_id, "REFERENCE", 0.95)  # Reflection references original bug
        
        flash("AI Coding Agent dataset loaded successfully! Complex bug fixing workflow with multiple attempts and solution paths created.", "success")
    except Exception as e:
        flash(f"Error loading example dataset: {e}", "error")
        
    return redirect(url_for('index'))

@app.errorhandler(500)
def server_error(e):
    flash('Server error', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8082)