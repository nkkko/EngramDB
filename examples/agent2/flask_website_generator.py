import os
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from flask import Flask, request, jsonify, render_template_string
from litellm import completion, set_verbose # Keep set_verbose if needed for debugging
import engramdb_py as engramdb

# Uncomment for debugging litellm issues
# set_verbose(True)

# Load environment variables from .env file if it exists
def load_env_file():
    env_path = Path('.env')
    if env_path.exists():
        print(f"Loading configuration from {env_path}")
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip('"\'')
                    if value and key not in os.environ:
                        os.environ[key] = value
                        print(f"  Set {key}")

load_env_file()

ENGRAMDB_PATH = os.environ.get("ENGRAMDB_PATH", "agent_memory.engramdb")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "generated_website")

# --- START FIX 1: Update CLAUDE_MODEL_MAPPING ---
CLAUDE_MODEL_MAPPING = {
    # Latest model names with litellm prefix
    "claude-3-7-sonnet-latest": "anthropic/claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-latest": "anthropic/claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-latest": "anthropic/claude-3-5-haiku-20241022",
    "claude-3-sonnet-latest": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku-latest": "anthropic/claude-3-haiku-20240307",
    "claude-3-opus-latest": "anthropic/claude-3-opus-20240229",

    # Handle names without 'latest' suffix
    "claude-3-7-sonnet": "anthropic/claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet-v2": "anthropic/claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet": "anthropic/claude-3-5-sonnet-20240620",
    "claude-3-5-haiku": "anthropic/claude-3-5-haiku-20241022",
    "claude-3-sonnet": "anthropic/claude-3-sonnet-20240229",
    "claude-3-haiku": "anthropic/claude-3-haiku-20240307",
    "claude-3-opus": "anthropic/claude-3-opus-20240229",
}
# --- END FIX 1 ---

# --- START FIX 2: Adjust Model Name Logic ---
MODEL_NAME = os.environ.get("LITELLM_MODEL", "")
if not MODEL_NAME:
    legacy_model = os.environ.get("PYDANTIC_AI_MODEL", "")
    if legacy_model:
        print(f"Found legacy PYDANTIC_AI_MODEL: {legacy_model}. Attempting conversion...")
        if ":" in legacy_model:
            provider, model = legacy_model.split(":", 1)
            provider = provider.lower() # Normalize provider name
            if provider == "openai":
                MODEL_NAME = model # Let litellm infer openai if no prefix
            elif provider == "anthropic":
                # Use mapping first, then add prefix if it looks like Claude
                mapped_name = CLAUDE_MODEL_MAPPING.get(model)
                if mapped_name:
                    MODEL_NAME = mapped_name
                elif "claude" in model.lower():
                    MODEL_NAME = f"anthropic/{model}"
                else: # Unrecognized anthropic model name
                    MODEL_NAME = f"anthropic/{model}" # Best guess prefix
            elif provider == "groq":
                MODEL_NAME = f"groq/{model}"
            # Add elif for other providers like google-gla, google-vertex etc. if needed
            else: # Unrecognized provider prefix in legacy var
                MODEL_NAME = legacy_model # Pass through, might fail
        else: # No provider prefix in legacy variable
             # Try mapping common Claude names first
             mapped_name = CLAUDE_MODEL_MAPPING.get(legacy_model)
             if mapped_name:
                 MODEL_NAME = mapped_name
             elif "claude" in legacy_model.lower(): # If it looks like Claude, add prefix
                 MODEL_NAME = f"anthropic/{legacy_model}"
             else: # Otherwise pass through
                 MODEL_NAME = legacy_model

        if MODEL_NAME and MODEL_NAME != legacy_model:
             print(f"  Converted to litellm format: {MODEL_NAME}")
        elif not MODEL_NAME:
             print(f"  Could not automatically convert legacy model name: {legacy_model}. Using it directly.")
             MODEL_NAME = legacy_model # Ensure MODEL_NAME is set even if conversion failed

# --- END FIX 2 ---

# --- START FIX 3: Refine Default Logic ---
# If MODEL_NAME is still not set, try defaults based on API keys
if not MODEL_NAME:
    print("No model specified via LITELLM_MODEL or PYDANTIC_AI_MODEL. Checking API keys for defaults...")
    if os.environ.get("OPENAI_API_KEY"):
        MODEL_NAME = "gpt-3.5-turbo"
        print("  Defaulting to OpenAI: gpt-3.5-turbo")
    elif os.environ.get("ANTHROPIC_API_KEY"):
        MODEL_NAME = "anthropic/claude-3-5-sonnet-20241022" # Latest Claude 3.5 Sonnet
        print(f"  Defaulting to Anthropic: {MODEL_NAME}")
    elif os.environ.get("GROQ_API_KEY"):
        MODEL_NAME = "groq/llama-3-70b-instruct"
        print(f"  Defaulting to Groq: {MODEL_NAME}")
    else:
        MODEL_NAME = "gpt-3.5-turbo" # Ultimate fallback
        print(f"  No specific API key found. Falling back to default: {MODEL_NAME}")
        print("  WARNING: This might fail if OPENAI_API_KEY is not set.")
# --- END FIX 3 ---

# --- START FIX 4: Remove Redundant Checks ---
# The following checks are now less reliable as MODEL_NAME should be correctly formatted.
# We can add a final sanity check instead.

# IS_CLAUDE_MODEL = "anthropic/" in MODEL_NAME.lower() # Check for prefix now

# Final sanity check for Claude models - ensure they have the anthropic/ prefix
if "claude" in MODEL_NAME.lower() and not MODEL_NAME.startswith("anthropic/"):
    # Check if it uses the old 'anthropic.' prefix and fix it
    if MODEL_NAME.startswith("anthropic."):
        correct_name = f"anthropic/{MODEL_NAME.split('.', 1)[1]}"
        print(f"Correcting Claude model name format: {MODEL_NAME} -> {correct_name}")
        MODEL_NAME = correct_name
    else:
        # Add the prefix if missing entirely
        correct_name = f"anthropic/{MODEL_NAME}"
        print(f"Adding missing prefix to Claude model name: {MODEL_NAME} -> {correct_name}")
        MODEL_NAME = correct_name

# Determine IS_CLAUDE_MODEL based on the final, corrected name
IS_CLAUDE_MODEL = MODEL_NAME.startswith("anthropic/")

# Remove the previous block that checked IS_CLAUDE_MODEL and tried to fix the name again.
# --- END FIX 4 ---


# Initialize Flask app
app = Flask(__name__)

# Memory types
MEMORY_TYPE_REQUIREMENT = "requirement"
MEMORY_TYPE_COMPONENT = "component"
MEMORY_TYPE_MESSAGE = "message"

# We'll use a consistent prefix for all generated routes
GENERATED_SITE_PREFIX = '/generated'
GENERATED_SITE_ENDPOINT_PREFIX = 'generated_site.'

@dataclass
class AgentContext:
    """Context object for the agent with database connection"""
    db: engramdb.Database
    chat_id: uuid.UUID = field(default_factory=uuid.uuid4)

    def text_to_vector(self, text: str) -> List[float]:
        """
        Create a simple vector embedding for text
        Note: In a real application, use a proper embedding model
        """
        text_bytes = text.encode('utf-8')
        hash_val = abs(hash(text_bytes)) % (2**32 - 1)
        np.random.seed(hash_val)
        return np.random.random(10).astype(np.float32).tolist()

    def store_memory(self, memory_type: str, content: Dict[str, Any]) -> str:
        """Store memory in EngramDB and return the node ID"""
        vector = self.text_to_vector(json.dumps(content))
        node = engramdb.MemoryNode(vector)

        node.set_attribute("memory_type", memory_type)
        node.set_attribute("timestamp", datetime.now().isoformat())
        node.set_attribute("chat_id", str(self.chat_id))

        for key, value in content.items():
            # Ensure value is serializable or handle appropriately
            try:
                # Convert UUID to string to avoid PyString conversion errors
                if isinstance(value, uuid.UUID):
                    value = str(value)
                    
                # Attempt basic serialization check, adjust if complex types needed
                json.dumps({key: value})
                node.set_attribute(key, value)
            except TypeError:
                print(f"Warning: Attribute '{key}' with value type '{type(value)}' might not be serializable for storage. Storing as string.")
                node.set_attribute(key, str(value))

        try:
            memory_id = self.db.save(node)
            return str(memory_id)
        except Exception as e:
            print(f"Error saving memory node: {e}")
            # Return a fallback ID in case of error
            return str(uuid.uuid4())

    def store_message(self, role: str, content: str) -> str:
        """Store a chat message in EngramDB"""
        return self.store_memory(MEMORY_TYPE_MESSAGE, {
            "role": role,
            "content": content
        })

    def store_requirement(self, requirement: str) -> str:
        """Store a user requirement in EngramDB"""
        return self.store_memory(MEMORY_TYPE_REQUIREMENT, {
            "content": requirement
        })

    def store_component(self, name: str, component_type: str, code: str, description: str) -> str:
        """Store a generated code component in EngramDB"""
        return self.store_memory(MEMORY_TYPE_COMPONENT, {
            "name": name,
            "type": component_type,
            "code": code,
            "description": description
        })

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

                if memory_type == MEMORY_TYPE_MESSAGE:
                    memory_data["role"] = node.get_attribute("role")
                    memory_data["content"] = node.get_attribute("content")
                elif memory_type == MEMORY_TYPE_REQUIREMENT:
                    memory_data["content"] = node.get_attribute("content")
                elif memory_type == MEMORY_TYPE_COMPONENT:
                    memory_data["name"] = node.get_attribute("name")
                    # Use 'component_type' consistently if that's the attribute name
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
                    if memory_type == MEMORY_TYPE_MESSAGE:
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
        try:
            memory_ids = self.db.list_all()
        except Exception as e:
            print(f"Error listing all memories when getting components: {e}")
            return []
            
        components = []

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
                    print(f"Skipping invalid memory ID format in components: {memory_id_bytes} (type: {type(memory_id_bytes)})")
                    continue
                
                if node_id is None:
                    continue
                
                try:
                    node = self.db.load(node_id)
                except Exception as load_error:
                    print(f"Error loading component node with ID {node_id}: {load_error}")
                    continue
                
                try:
                    memory_type = node.get_attribute("memory_type")
                    if memory_type == MEMORY_TYPE_COMPONENT:
                        components.append({
                            "id": str(node_id),
                            "name": node.get_attribute("name"),
                            "type": node.get_attribute("type"),
                            "code": node.get_attribute("code"),
                            "description": node.get_attribute("description"),
                            "timestamp": node.get_attribute("timestamp")
                        })
                except Exception as attr_error:
                    print(f"Error getting component attributes from node {node_id}: {attr_error}")
                    continue
            except Exception as e:
                print(f"Error processing component node: {e}")

        return components


def get_system_prompt(context: AgentContext) -> str:
    """Generate a system prompt with relevant context from EngramDB"""
    recent_messages = context.get_chat_history(5)
    message_history = "\n".join([
        f"{msg['role'].upper()}: {msg['content']}"
        for msg in recent_messages
    ])

    components = context.get_all_components()
    component_list = "\n".join([
        f"- {comp['name']} ({comp['type']}): {comp['description']}"
        for comp in components
    ])

    prompt = f"""You are an expert Flask website generator that helps users build web applications.

CONTEXT FROM MEMORY:
===================
Recent conversation:
{message_history}

Existing components:
{component_list if components else "No components generated yet."}
===================

When users request features or describe requirements, identify them clearly and create appropriate Flask components.
Each component should be complete and follow best practices.

IMPORTANT: You can generate these Flask file types:
1. Main application (app.py)
2. Routes (Python files with Flask routes)
3. Templates (HTML files with Jinja2)
4. Static files (CSS or JavaScript)
5. Models (database models)
6. Forms (WTForms for Flask)

Keep all components modular, well-organized, and maintainable."""

    if IS_CLAUDE_MODEL:
        prompt += """

AVAILABLE COMMANDS:
You can use the following commands by writing them in your response:

1. /store_requirement {requirement}
   Stores a user requirement for future reference
   Example: /store_requirement The website needs user authentication

2. /store_component {name} {type} {description}
   {code}
   /end_component
   Stores a generated code component
   Example:
   /store_component main main The main Flask application
   from flask import Flask, render_template

   app = Flask(__name__)

   @app.route('/')
   def home():
       return render_template('index.html')

   if __name__ == '__main__':
       app.run(debug=True)
   /end_component

3. /search_similar {query} {limit}
   Search for similar memories (optional limit parameter)
   Example: /search_similar user authentication 3

4. /save_files
   Save all components to disk
   Example: /save_files

When using commands, make sure to follow the exact format. I'll process these commands and store the information in EngramDB.
"""
    else:
        prompt += """
USE THE FUNCTION CALLING CAPABILITY to store new requirements, create website components, and search for relevant context.
"""

    return prompt


# Store global route state to avoid re-registering routes
_ROUTE_STATE = {
    'registered_routes': set(),  # Set of route paths that have been registered
    'registered_templates': {},  # Dictionary of registered templates
    'registered_statics': {},    # Dictionary of registered static files
    'registered_routes_funcs': {}, # Dictionary of registered route functions
    'route_modules': []          # List of imported route modules
}

def register_generated_routes(context: AgentContext):
    """
    Dynamically register route handlers for the generated website
    This makes the generated website available at /generated path
    """
    global _ROUTE_STATE
    components = context.get_all_components()
    
    from flask import render_template_string, send_file, Response
    from io import BytesIO
    import importlib.util
    import sys
    import tempfile
    
    # Store templates, static files, and routes for dynamic serving
    templates = _ROUTE_STATE['registered_templates']
    static_files = _ROUTE_STATE['registered_statics']
    route_files = {}
    
    print(f"Registering dynamic routes for {len(components)} components...")
    
    # Get routes for display
    def get_routes():
        routes = []
        for rule in app.url_map.iter_rules():
            if (rule.endpoint.startswith('generated_site.') and 
                'static' not in rule.endpoint and 
                not str(rule.rule).endswith('.css') and 
                not str(rule.rule).endswith('.js')):
                # Replace the /generated prefix to make relative links work properly
                route = str(rule.rule).replace('/generated', '')
                if route and route != '/':
                    routes.append(route)
        return routes
    
    # Register home route only if not already registered
    if '/' not in _ROUTE_STATE['registered_routes']:
        try:
            # Using app.add_url_rule instead of decorators to avoid the "already registered" error
            def generated_home():
                routes = get_routes()
                return render_template_string('''
                    <html>
                    <head>
                        <title>Generated Flask Website</title>
                        <style>
                            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                            pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; }
                            .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
                        </style>
                    </head>
                    <body>
                        <h1>Generated Flask Website</h1>
                        <p>This is your dynamically generated Flask application.</p>
                        
                        {% if routes %}
                        <div class="card">
                            <h3>Available Routes:</h3>
                            <ul>
                            {% for route in routes %}
                                <li><a href="{{ route }}">{{ route }}</a></li>
                            {% endfor %}
                            </ul>
                        </div>
                        {% endif %}
                        
                        {% if components %}
                        <div class="card">
                            <h3>Generated Components:</h3>
                            <ul>
                            {% for comp in components %}
                                <li><strong>{{ comp.name }}</strong> ({{ comp.type }}): {{ comp.description }}</li>
                            {% endfor %}
                            </ul>
                        </div>
                        {% endif %}
                    </body>
                    </html>
                ''', routes=routes, components=components)
            
            app.add_url_rule('/generated/', endpoint='generated_site.generated_home', view_func=generated_home)
            _ROUTE_STATE['registered_routes'].add('/')
            _ROUTE_STATE['registered_routes_funcs']['generated_home'] = generated_home
            print("Registered home route for generated site")
        except Exception as e:
            print(f"Error registering home route: {e}")
    
    # Process each component
    for component in components:
        name = component["name"]
        comp_type = component["type"]
        code = component["code"]
        
        try:
            if comp_type == "template":
                templates[name] = code
                route_path = f'/templates/{name}.html'
                
                if route_path not in _ROUTE_STATE['registered_routes']:
                    # Need to define a unique function name for each route
                    route_func_name = f"serve_template_{name}_{hash(code) % 10000}"
                    
                    # Create a handler function for this template
                    def template_handler(template_content=code):
                        return render_template_string(template_content)
                    
                    # Add the route to the Flask app directly
                    app.add_url_rule(
                        f'/generated{route_path}', 
                        endpoint=f'generated_site.{route_func_name}',
                        view_func=template_handler
                    )
                    _ROUTE_STATE['registered_routes'].add(route_path)
                    _ROUTE_STATE['registered_routes_funcs'][route_func_name] = template_handler
                    print(f"Registered template route: {route_path}")
            
            elif comp_type == "static":
                # Handle different static file types
                if "css" in name.lower():
                    mimetype = "text/css"
                    static_files[f"{name}.css"] = (code, mimetype)
                    route_path = f'/static/css/{name}.css'
                elif "js" in name.lower() or "javascript" in name.lower():
                    mimetype = "application/javascript"
                    static_files[f"{name}.js"] = (code, mimetype)
                    route_path = f'/static/js/{name}.js'
                else:
                    mimetype = "text/plain"
                    static_files[f"{name}.txt"] = (code, mimetype)
                    route_path = f'/static/{name}.txt'
                
                if route_path not in _ROUTE_STATE['registered_routes']:
                    # Need to define a unique function name for each route
                    route_func_name = f"serve_static_{name}_{hash(code) % 10000}"
                    
                    # Create a handler function for this static file
                    def static_handler(content=code, content_type=mimetype):
                        return Response(content, mimetype=content_type)
                    
                    # Add the route to the Flask app directly
                    app.add_url_rule(
                        f'/generated{route_path}',
                        endpoint=f'generated_site.{route_func_name}',
                        view_func=static_handler
                    )
                    _ROUTE_STATE['registered_routes'].add(route_path)
                    _ROUTE_STATE['registered_routes_funcs'][route_func_name] = static_handler
                    print(f"Registered static route: {route_path}")
            
            elif comp_type == "route":
                route_files[name] = code
        
        except Exception as e:
            print(f"Error processing component '{name}' ({comp_type}): {e}")
    
    # Execute route code to register route handlers
    if route_files:
        # Create a temporary directory for route files
        temp_dir = tempfile.mkdtemp()
        sys.path.insert(0, temp_dir)
        
        # Create __init__.py to make it a package
        with open(f"{temp_dir}/__init__.py", "w") as f:
            f.write("")
        
        # Write route files to the temporary directory
        for name, code in route_files.items():
            if name in _ROUTE_STATE['registered_routes_funcs']:
                print(f"Route module '{name}' already registered, skipping")
                continue
                
            file_path = f"{temp_dir}/{name}.py"
            with open(file_path, "w") as f:
                # Modify the route registration to work with Flask directly
                route_code = f"""
from flask import Blueprint, render_template, request, redirect, url_for, flash, jsonify, session
import sys
sys.path.append('{__file__}')

# Import the Flask app directly
from flask_website_generator import app

# Modified route code for direct registration with Flask
{code}

# Function to register routes directly with Flask app
def register_routes():
    # Find all the route functions in this module and register them
    import inspect
    from flask import Flask
    
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if hasattr(obj, '__wrapped__') and hasattr(obj.__wrapped__, 'route_info'):
            route_info = obj.__wrapped__.route_info
            for route_path, options in route_info:
                full_path = '/generated' + route_path
                endpoint = 'generated_site.' + name + '_' + str(hash(route_path) % 10000)
                app.add_url_rule(full_path, endpoint=endpoint, view_func=obj, **options)
                print(f"Registered custom route: {{route_path}}")

# Helper decorator to store route info for registration
def route(rule, **options):
    def decorator(f):
        if not hasattr(f, 'route_info'):
            f.route_info = []
        f.route_info.append((rule, options))
        return f
    return decorator

# Call register_routes to add all routes to the app
register_routes()
"""
                f.write(route_code)
            
            # Import the route module
            try:
                module_name = name
                spec = importlib.util.spec_from_file_location(module_name, file_path)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                _ROUTE_STATE['route_modules'].append(module)
                _ROUTE_STATE['registered_routes_funcs'][name] = module
                print(f"Successfully imported route module '{name}'")
            except Exception as e:
                print(f"Error importing route module '{name}': {e}")
    
    # Return available routes for display
    routes = []
    for rule in app.url_map.iter_rules():
        if (rule.endpoint.startswith('generated_site.') and 
            'static' not in rule.endpoint and 
            not str(rule.rule).endswith('.css') and 
            not str(rule.rule).endswith('.js')):
            routes.append(rule.rule)
    
    return routes

def save_files_to_disk(context: AgentContext) -> Dict[str, Any]:
    """Save all components to disk"""
    components = context.get_all_components()

    if not components:
        return {"success": False, "message": "No components to save", "files": []}

    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "templates").mkdir(exist_ok=True)
    (output_path / "static").mkdir(exist_ok=True)
    (output_path / "static" / "css").mkdir(exist_ok=True)
    (output_path / "static" / "js").mkdir(exist_ok=True)

    saved_files = []

    for component in components:
        name = component["name"]
        comp_type = component["type"]
        code = component["code"]
        filepath = None # Initialize filepath

        try:
            # Determine the file path based on component type
            if comp_type == "route" or comp_type == "main":
                filename = "app.py" if name == "main" else f"{name}.py"
                filepath = output_path / filename
            elif comp_type == "template":
                filename = f"{name}.html"
                filepath = output_path / "templates" / filename
            elif comp_type == "static":
                # Basic check for css/js, default to .txt if unsure
                if "css" in name.lower():
                    filename = f"{name}.css"
                    filepath = output_path / "static" / "css" / filename
                elif "js" in name.lower() or "javascript" in name.lower():
                    filename = f"{name}.js" 
                    filepath = output_path / "static" / "js" / filename
                else:
                    filename = f"{name}.txt" # Fallback
                    filepath = output_path / "static" / filename
            elif comp_type == "model":
                filename = "models.py" # Usually keep models together
                filepath = output_path / filename
            elif comp_type == "form":
                filename = "forms.py" # Usually keep forms together
                filepath = output_path / filename
            else:
                # Default for unknown types or when name suggests extension
                if '.' in name:
                    filename = name
                else:
                    filename = f"{name}.py" # Assume python if no extension hint
                filepath = output_path / filename

            # Ensure directory exists if nested (e.g., static/css/style.css)
            filepath.parent.mkdir(parents=True, exist_ok=True)

            # Write the file
            with open(filepath, 'w') as f:
                f.write(code)

            saved_files.append(str(filepath.relative_to(Path.cwd()))) # Store relative path
        except Exception as e:
            print(f"Error saving component '{name}' ({comp_type}) to {filepath}: {e}")


    # Create a requirements.txt file if it doesn't exist or append flask
    req_path = output_path / "requirements.txt"
    current_reqs = set()
    if req_path.exists():
        with open(req_path, 'r') as f:
            current_reqs = {line.strip() for line in f if line.strip()}

    required = {"flask>=2.0.0", "flask-wtf>=1.0.0", "flask-sqlalchemy>=3.0.0"}
    needs_update = False
    for req in required:
        pkg_name = req.split(">=")[0].split("==")[0].split("<")[0] # Basic package name extraction
        if not any(existing.startswith(pkg_name) for existing in current_reqs):
            current_reqs.add(req)
            needs_update = True

    if needs_update:
        with open(req_path, 'w') as f:
            f.write("\n".join(sorted(list(current_reqs))))
        if str(req_path.relative_to(Path.cwd())) not in saved_files:
             saved_files.append(str(req_path.relative_to(Path.cwd())))
    
    # Register dynamic routes for the generated application
    print("Registering dynamic routes from save_files_to_disk...")
    register_generated_routes(context)
    
    # Register the blueprint with the main app if not already registered
    try:
        if 'generated_site' not in app.blueprints:
            print("Registering blueprint with the Flask app...")
            app.register_blueprint(generated_site)
    except AssertionError as e:
        print(f"Blueprint registration error (application already started): {e}")
        print("Blueprint already registered or app has already handled its first request.")

    return {
        "success": True,
        "message": f"Saved/Updated {len(saved_files)} files in {output_path} and registered dynamic routes",
        "files": saved_files
    }


# Global context to reuse across requests
_GLOBAL_CONTEXT = None

def initialize_agent_context() -> AgentContext:
    """Initialize the agent context with EngramDB database, reusing global instance if available"""
    global _GLOBAL_CONTEXT
    
    # Return the existing context if already initialized
    if _GLOBAL_CONTEXT is not None:
        return _GLOBAL_CONTEXT
    
    # Try to create a fresh database if there are issues with the existing one
    try_new_db = False
    db_path = Path(ENGRAMDB_PATH)
    
    # Check if the existing database has problems
    if db_path.exists() and db_path.stat().st_size > 0:
        try:
            # Test if we can open and use the existing database
            test_db = engramdb.Database.file_based(str(db_path))
            # Test a simple operation
            test_db.list_all()
            print(f"Using existing database at: {db_path.resolve()}")
        except Exception as test_error:
            print(f"Error accessing existing database at '{ENGRAMDB_PATH}': {test_error}")
            print("The database file might be corrupted or incompatible.")
            # Create backup of the problematic database
            backup_path = f"{ENGRAMDB_PATH}.bak.{int(datetime.now().timestamp())}"
            try:
                import shutil
                shutil.copy2(db_path, backup_path)
                print(f"Created backup of problematic database at: {backup_path}")
                # Remove the problematic database
                db_path.unlink()
                print(f"Removed problematic database. Will create a fresh one.")
                try_new_db = True
            except Exception as backup_error:
                print(f"Error backing up problematic database: {backup_error}")
                # Will still try to create a new in-memory database
    
    try:
        # Ensure parent directory exists if db_path includes directories
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        if try_new_db or not db_path.exists():
            # Fresh database needed
            db = engramdb.Database.file_based(str(db_path))
            print(f"Created new database at: {db_path.resolve()}")
        else:
            # Try using the existing database
            db = engramdb.Database.file_based(str(db_path))
            print(f"Using database at: {db_path.resolve()}")
            
        _GLOBAL_CONTEXT = AgentContext(db=db)
        return _GLOBAL_CONTEXT
    except Exception as e:
        print(f"Error initializing file-based database at '{ENGRAMDB_PATH}': {e}")
        print("Falling back to in-memory database.")
        try:
            db = engramdb.Database.in_memory()
            _GLOBAL_CONTEXT = AgentContext(db=db)
            return _GLOBAL_CONTEXT
        except Exception as mem_error:
            print(f"Error creating in-memory database: {mem_error}")
            # Last resort - create a dummy context with minimal functionality
            from types import SimpleNamespace
            dummy_db = SimpleNamespace()
            dummy_db.list_all = lambda: []
            dummy_db.save = lambda _: uuid.uuid4()
            dummy_db.load = lambda _: SimpleNamespace(get_attribute=lambda _: "")
            dummy_db.search_similar = lambda *args, **kwargs: []
            dummy_context = AgentContext(db=dummy_db)
            _GLOBAL_CONTEXT = dummy_context
            return dummy_context


# Define the tools available to the agent (for function-calling models)
tools = [
    {
        "type": "function",
        "function": {
            "name": "store_requirement",
            "description": "Store a user requirement in EngramDB for future reference",
            "parameters": {
                "type": "object",
                "properties": {
                    "requirement": {
                        "type": "string",
                        "description": "The user requirement to store"
                    }
                },
                "required": ["requirement"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "store_component",
            "description": "Store a generated website component in EngramDB",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the component (e.g., 'login_form', 'user_model')"
                    },
                    "component_type": {
                        "type": "string",
                        "description": "Type of component (route, template, static, model, form, main)"
                    },
                    "code": {
                        "type": "string",
                        "description": "The complete code for this component"
                    },
                    "description": {
                        "type": "string",
                        "description": "Brief description of what this component does"
                    }
                },
                "required": ["name", "component_type", "code", "description"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_similar",
            "description": "Search for similar memories in EngramDB",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query text"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum number of results to return"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_files",
            "description": "Save all components to disk",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    }
]


def check_api_key_setup() -> bool:
    """Check if API keys are properly set up"""
    has_openai = os.environ.get("OPENAI_API_KEY") is not None
    has_anthropic = os.environ.get("ANTHROPIC_API_KEY") is not None
    has_groq = os.environ.get("GROQ_API_KEY") is not None

    return has_openai or has_anthropic or has_groq


def create_example_env_file():
    """Create an example .env file if it doesn't exist"""
    env_path = Path('.env.example')
    if not env_path.exists():
        example_content = """# EngramDB Agent Configuration
# Uncomment and set values for the LLM provider you want to use

# LLM Provider Options (uncomment one)
# LITELLM_MODEL=gpt-3.5-turbo
# LITELLM_MODEL=gpt-4o
# LITELLM_MODEL=anthropic/claude-3-haiku-20240307
# LITELLM_MODEL=anthropic/claude-3-sonnet-20240229
# LITELLM_MODEL=anthropic/claude-3-opus-20240229
# LITELLM_MODEL=anthropic/claude-3-5-sonnet-20240620
# LITELLM_MODEL=groq/llama-3-8b-8192

# API Keys (uncomment and set the one corresponding to your chosen model)
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here
# GROQ_API_KEY=your_groq_key_here

# Storage paths
# ENGRAMDB_PATH=agent_memory.engramdb
# OUTPUT_PATH=generated_website
"""
        with open(env_path, 'w') as f:
            f.write(example_content)
        print(f"Created example environment file at {env_path}")
        print("Copy this file to .env and update with your API keys.")


# Function to parse Claude's command syntax from text
def parse_commands_from_text(text: str, context: AgentContext) -> List[Dict[str, Any]]:
    """Parse commands from text for models that don't support function calling"""
    tool_calls = []

    # Regex patterns
    req_pattern = r'/store_requirement\s+(.*?)(?=(?:/\w+)|$)'
    comp_pattern = r'/store_component\s+(\S+)\s+(\S+)\s+(.*?)\n(.*?)/end_component'
    search_pattern = r'/search_similar\s+([^/\n]+)(?:\s+(\d+))?'
    save_pattern = r'/save_files'

    # Find all command matches using non-overlapping search
    all_matches = []
    last_end = 0
    while last_end < len(text):
        found_match = False
        for pattern, cmd_type in [(req_pattern, "store_requirement"),
                                 (comp_pattern, "store_component"),
                                 (search_pattern, "search_similar"),
                                 (save_pattern, "save_files")]:
            match = re.search(pattern, text[last_end:], re.DOTALL | re.IGNORECASE)
            if match:
                all_matches.append((match.start() + last_end, match.end() + last_end, cmd_type, match.groups()))
                found_match = True
                break # Prioritize first found command from current position
        if not found_match:
            break # No more commands found
        last_end = match.end() + last_end if match else len(text)

    # Sort matches by starting position
    all_matches.sort(key=lambda x: x[0])

    # Process matches
    for _, _, cmd_type, groups in all_matches:
        try:
            if cmd_type == "store_requirement":
                requirement = groups[0].strip()
                if requirement:
                    memory_id = context.store_requirement(requirement)
                    tool_calls.append({"tool": "store_requirement", "result": f"Requirement stored with ID: {memory_id}"})
            elif cmd_type == "store_component":
                name, comp_type, description, code = groups
                name = name.strip()
                comp_type = comp_type.strip()
                description = description.strip()
                code = code.strip()
                if name and comp_type and code:
                    memory_id = context.store_component(name, comp_type, code, description)
                    tool_calls.append({"tool": "store_component", "result": f"Component '{name}' stored with ID: {memory_id}"})
            elif cmd_type == "search_similar":
                 query = groups[0].strip()
                 limit_str = groups[1]
                 limit = int(limit_str) if limit_str and limit_str.isdigit() else 5
                 if query:
                     results = context.search_similar(query, limit)
                     tool_calls.append({"tool": "search_similar", "result": results})
            elif cmd_type == "save_files":
                 result = save_files_to_disk(context)
                 tool_calls.append({"tool": "save_files", "result": result})
        except Exception as e:
            print(f"Error processing command {cmd_type}: {e}")

    return tool_calls


def process_user_message(user_message: str, context: AgentContext) -> Dict[str, Any]:
    """Process a user message with the LLM agent"""
    context.store_message("user", user_message)

    if user_message.lower() == "save":
        save_result = save_files_to_disk(context)
        context.store_message("system", f"Files saved: {save_result['message']}")
        return {"response": save_result["message"], "files_saved": save_result["files"]}

    if not check_api_key_setup():
        api_key_message = (
             "Error: No API key found for any supported LLM provider. "
             "Please set at least one of these environment variables:\n"
             "- OPENAI_API_KEY\n- ANTHROPIC_API_KEY\n- GROQ_API_KEY\n\n"
             "Or create a .env file."
        )
        create_example_env_file()
        context.store_message("system", api_key_message)
        return {"response": api_key_message, "error": "missing_api_key"}

    try:
        system_prompt = get_system_prompt(context)
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
        kwargs = {} # Reset kwargs for each call

        # Set specific kwargs only if needed (e.g., for Anthropic max_tokens)
        if IS_CLAUDE_MODEL:
             kwargs["max_tokens"] = 4000
             # No need to set api_key here, litellm handles it via env var

        # Call completion
        if IS_CLAUDE_MODEL:
            print(f"Calling Claude model ({MODEL_NAME}) without function calling...")
            response = completion(model=MODEL_NAME, messages=messages, **kwargs)
        else:
            print(f"Calling model ({MODEL_NAME}) with function calling...")
            response = completion(model=MODEL_NAME, messages=messages, tools=tools, tool_choice="auto", **kwargs)

        # Extract assistant message and usage
        assistant_message = response.choices[0].message
        content = assistant_message.content or ""
        llm_usage = response.usage # Capture usage info

        # Store assistant message
        context.store_message("assistant", content)

        # Process tool calls or parse commands
        tool_results = []
        if IS_CLAUDE_MODEL:
            tool_results = parse_commands_from_text(content, context)
        elif hasattr(assistant_message, 'tool_calls') and assistant_message.tool_calls:
            for tool_call in assistant_message.tool_calls:
                function_name = tool_call.function.name
                try:
                    function_args = json.loads(tool_call.function.arguments)
                except json.JSONDecodeError:
                     print(f"Warning: Could not decode arguments for tool {function_name}: {tool_call.function.arguments}")
                     continue # Skip this tool call if args are invalid JSON

                result = None
                if function_name == "store_requirement":
                    requirement = function_args.get("requirement")
                    if requirement: memory_id = context.store_requirement(requirement); result = f"Requirement stored with ID: {memory_id}"
                elif function_name == "store_component":
                    name = function_args.get("name")
                    component_type = function_args.get("component_type")
                    code = function_args.get("code")
                    description = function_args.get("description")
                    if all([name, component_type, code, description]):
                         memory_id = context.store_component(name, component_type, code, description)
                         result = f"Component '{name}' stored with ID: {memory_id}"
                elif function_name == "search_similar":
                    query = function_args.get("query")
                    limit = function_args.get("limit", 5)
                    if query: result = context.search_similar(query, limit)
                elif function_name == "save_files":
                    result = save_files_to_disk(context)

                if result is not None:
                     tool_results.append({"tool": function_name, "result": result})
                else:
                     print(f"Warning: Tool '{function_name}' called but execution failed or produced no result.")

        return {
            "response": content,
            "tool_calls": tool_results,
            "usage": { # Include usage info
                "prompt_tokens": llm_usage.prompt_tokens,
                "completion_tokens": llm_usage.completion_tokens,
                "total_tokens": llm_usage.total_tokens,
            } if llm_usage else None
        }
    except Exception as e:
        error_message = f"Error during LLM call: {str(e)}"
        import traceback
        trace = traceback.format_exc()
        print(f"DETAILED ERROR: {trace}")
        context.store_message("system", error_message)
        return {
            "response": f"I encountered an error: {str(e)}\n\nPlease make sure your API keys are correctly set up and the model '{MODEL_NAME}' is accessible.",
            "error": str(e)
        }

# --- Web Server Integration ---

# CLI interface
def run_cli():
    """Run the CLI interface for the website generator"""
    print("\n🌐 EngramDB Flask Website Generator 🌐\n")
    print("This agent helps you build Flask websites with EngramDB for memory.")
    print("Type 'exit' to quit, 'save' to save files to disk.\n")

    if not check_api_key_setup():
        print("⚠️  No API key configured. Please set up at least one of these environment variables:")
        print("- OPENAI_API_KEY\n- ANTHROPIC_API_KEY\n- GROQ_API_KEY\n")
        create_example_env_file()
    else:
        print(f"Using model: {MODEL_NAME}\n")
        if IS_CLAUDE_MODEL:
            print("Note: Using Claude model with command parsing instead of function calling.\n")

    context = initialize_agent_context()

    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError:
            print("\nExiting...")
            break


        if user_input.lower() in ['exit', 'quit']:
            print("\nThank you for using the Flask Website Generator. Goodbye!")
            break

        print("\nThinking...")
        response = process_user_message(user_input, context)

        if "error" in response and response.get("error") == "missing_api_key":
            print("\n" + response["response"])
            continue
        elif "error" in response:
             print(f"\nAgent Error: {response['response']}") # Display specific error
             continue # Allow user to try again or exit

        print(f"\nAgent: {response['response']}")

        # Show tool calls if any
        if response.get("tool_calls"):
            print("\nActions taken:")
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["tool"]
                result = tool_call["result"]

                if tool_name == "store_component":
                    print(f"  - Stored component: {result}")
                elif tool_name == "store_requirement":
                    print(f"  - Stored requirement: {result}")
                elif tool_name == "save_files":
                    if isinstance(result, dict) and result.get("success"):
                        print(f"  - {result['message']}")
                        if result.get("files"):
                            print("    Files:")
                            for f in result["files"]: print(f"      - {f}")
                    else:
                        print(f"  - Save command result: {result.get('message', 'Failed to save')}")
                elif tool_name == "search_similar":
                     print(f"  - Searched for '{tool_call.get('query','')}': Found {len(result)} results") # Be more informative

        # Show usage if available
        if response.get("usage"):
            usage = response["usage"]
            print(f"\nUsage: Prompt Tokens={usage.get('prompt_tokens', 'N/A')}, Completion Tokens={usage.get('completion_tokens', 'N/A')}, Total Tokens={usage.get('total_tokens', 'N/A')}")


# Web interface routes
@app.route('/', methods=['GET', 'POST'])
def home():
    """Home page - displays a form for user input or progress"""
    # Initialize context to check if website has been generated
    context = initialize_agent_context()
    has_components = len(context.get_all_components()) > 0
    
    if request.method == 'POST':
        user_input = request.form.get('user_input', '')
        
        # Handle the 'save' shortcut for web interface
        if user_input.lower() == 'save':
            result = {
                "response": "Saving all components to disk...",
                "tool_calls": [{"tool": "save_files", "result": save_files_to_disk(context)}],
                "error": None
            }
        else:
            # Normal message processing
            result = process_user_message(user_input, context)

        components_created = [tc["result"] for tc in result.get("tool_calls", []) if tc["tool"] == "store_component"]
        files_saved_result = next((tc["result"] for tc in result.get("tool_calls", []) if tc["tool"] == "save_files"), None)
        files_saved_list = files_saved_result.get("files", []) if isinstance(files_saved_result, dict) else []
        save_message = files_saved_result.get("message", "") if isinstance(files_saved_result, dict) else ""
        
        # Check if new components were created, just register routes (blueprint is already registered)
        if components_created:
            try:
                # Register routes for the newly created components
                print("Registering dynamic routes for new components...")
                register_generated_routes(context)
                has_components = True
            except Exception as e:
                print(f"Error registering routes for new components: {e}")
                # Just continue without stopping execution
                pass

        return render_template_string('''
            <html>
            <head>
                <title>Flask Website Generator</title>
                <style>
                    body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                    pre { background: #f4f4f4; padding: 10px; border-radius: 5px; overflow-x: auto; white-space: pre-wrap; word-wrap: break-word; }
                    .card { border: 1px solid #ddd; border-radius: 5px; padding: 15px; margin-bottom: 15px; }
                    .error { background: #ffebee; padding: 15px; border-radius: 5px; border-left: 5px solid #f44336; }
                    .btn { display: inline-block; padding: 10px 15px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; }
                    .btn:hover { background: #45a049; }
                    .btn-secondary { background: #2196F3; }
                    .btn-secondary:hover { background: #0b7dda; }
                </style>
            </head>
            <body>
                <h1>Flask Website Generator</h1>

                {% if error %}
                <div class="error">
                    <p>{{ response }}</p>
                </div>
                {% else %}
                 <div class="card">
                     <h3>Your Request:</h3>
                     <p>{{ user_input }}</p>
                 </div>
                 <div class="card">
                     <h3>Agent Response:</h3>
                     <pre>{{ response }}</pre> {# Use pre for better formatting #}
                 </div>
                 {% if components_created %}
                 <div class="card">
                     <h3>Components Stored:</h3>
                     <ul>
                     {% for component_info in components_created %}
                         <li>{{ component_info }}</li>
                     {% endfor %}
                     </ul>
                 </div>
                 {% endif %}
                 {% if files_saved_list %}
                 <div class="card">
                     <h3>{{ save_message }}</h3>
                     <ul>
                     {% for file in files_saved_list %}
                         <li>{{ file }}</li>
                     {% endfor %}
                     </ul>
                 </div>
                 {% endif %}
                 
                 <div style="margin-top: 20px;">
                    <a href="/" class="btn">Ask something else</a>
                    {% if has_website %}
                    <a href="/generated" class="btn btn-secondary">View Generated Website</a>
                    {% endif %}
                 </div>
                {% endif %}

            </body>
            </html>
        ''', user_input=user_input, response=result["response"],
            components_created=components_created, files_saved_list=files_saved_list,
            save_message=save_message, error=result.get("error"), has_website=has_components)
    else:
        api_key_warning = ""
        if not check_api_key_setup():
            create_example_env_file() # Ensure example exists
            api_key_warning = '''
                <div style="background: #fff8e1; padding: 15px; border-radius: 5px; border-left: 5px solid #ffc107; margin-bottom: 20px;">
                    <h3>⚠️ API Key Required</h3>
                    <p>No API key configured. Please set up at least one of these environment variables:</p>
                    <ul><li>OPENAI_API_KEY</li><li>ANTHROPIC_API_KEY</li><li>GROQ_API_KEY</li></ul>
                    <p>Or create a .env file (example created as .env.example).</p>
                </div>
            '''
        
        # Check if we have a generated website to show
        website_section = ""
        if has_components:
            try:
                # Just register routes (the blueprint is already registered at module load time)
                register_generated_routes(context)
            except Exception as e:
                print(f"Error registering routes from home route: {e}")
                # Continue without stopping execution
            
            website_section = '''
                <div style="background: #e8f5e9; padding: 15px; border-radius: 5px; border-left: 5px solid #4CAF50; margin-bottom: 20px;">
                    <h3>✅ Generated Website Available</h3>
                    <p>Your previously generated website is available at: 
                       <a href="/generated" style="color: #2196F3; font-weight: bold;">/generated</a>
                    </p>
                </div>
            '''

        return render_template_string('''
            <html>
            <head>
                <title>Flask Website Generator</title>
                 <style>
                     body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
                     textarea { width: 100%; height: 150px; padding: 10px; margin-bottom: 10px; border: 1px solid #ccc; border-radius: 4px; font-size: 1em; }
                     button { padding: 10px 20px; background: #4CAF50; color: white; border: none; cursor: pointer; border-radius: 4px; font-size: 1em; }
                     button:hover { background: #45a049; }
                     .btn-secondary { display: inline-block; padding: 10px 15px; background: #2196F3; color: white; text-decoration: none; border-radius: 4px; margin-left: 10px; }
                     .btn-secondary:hover { background: #0b7dda; }
                 </style>
            </head>
            <body>
                <h1>EngramDB Flask Website Generator</h1>
                {{ api_key_warning|safe }}
                {{ website_section|safe }}
                <p>Describe the Flask app you want to create (or type 'save'):</p>
                <form method="post">
                    <textarea name="user_input" placeholder="E.g., I need a simple blog with user authentication..."></textarea>
                    <div style="display: flex; align-items: center;">
                        <button type="submit">Generate</button>
                        {% if has_website %}
                        <a href="/generated" class="btn-secondary">View Generated Website</a>
                        {% endif %}
                    </div>
                </form>
            </body>
            </html>
        ''', api_key_warning=api_key_warning, website_section=website_section, has_website=has_components)

@app.route('/api/message', methods=['POST'])
def api_message():
    """API endpoint for message processing"""
    data = request.json
    user_message = data.get('message', '')

    # Use global context
    context = initialize_agent_context()
    
    # Handle the 'save' command
    if user_message.lower() == 'save':
        result = {
            "response": "Saving all components to disk...",
            "tool_calls": [{"tool": "save_files", "result": save_files_to_disk(context)}],
            "error": None
        }
    else:
        result = process_user_message(user_message, context)

    return jsonify(result)


def run_integrated_server():
    """Start integrated server that serves both the CLI and web interface"""
    from threading import Thread
    import sys
    
    # Initialize context and register dynamic routes if any components exist
    context = initialize_agent_context()
    components = context.get_all_components()
    if components:
        print(f"Found {len(components)} existing components. Registering dynamic routes...")
        register_generated_routes(context)
    
    # Start the Flask server in a separate thread
    def start_flask_server():
        print(f"Starting web interface at http://127.0.0.1:8080 for model {MODEL_NAME}")
        print(f"Generated website will be available at http://127.0.0.1:8080/generated")
        # Set use_reloader=False to avoid duplicate processes
        app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)
    
    # Start Flask in a thread
    flask_thread = Thread(target=start_flask_server)
    flask_thread.daemon = True  # Thread will exit when main thread exits
    flask_thread.start()
    
    # Run the CLI in the main thread
    print("\n🌐 EngramDB Flask Website Generator 🌐\n")
    print("This agent helps you build Flask websites with EngramDB for memory.")
    print("Type 'exit' to quit, 'save' to save files to disk.\n")
    print(f"Using model: {MODEL_NAME}\n")
    print("Web interface is running at http://127.0.0.1:8080")
    if components:
        print("Your generated website is running at http://127.0.0.1:8080/generated\n")
    
    if IS_CLAUDE_MODEL:
        print("Note: Using Claude model with command parsing instead of function calling.\n")
    
    while True:
        try:
            user_input = input("\nYou: ")
        except EOFError:
            print("\nExiting...")
            break
        
        if user_input.lower() in ['exit', 'quit']:
            print("\nThank you for using the Flask Website Generator. Goodbye!")
            break
        
        print("\nThinking...")
        response = process_user_message(user_input, context)
        
        if "error" in response and response.get("error") == "missing_api_key":
            print("\n" + response["response"])
            continue
        elif "error" in response:
             print(f"\nAgent Error: {response['response']}") # Display specific error
             continue # Allow user to try again or exit
        
        print(f"\nAgent: {response['response']}")
        
        # Show tool calls if any
        if response.get("tool_calls"):
            print("\nActions taken:")
            for tool_call in response["tool_calls"]:
                tool_name = tool_call["tool"]
                result = tool_call["result"]
                
                if tool_name == "store_component":
                    print(f"  - Stored component: {result}")
                elif tool_name == "store_requirement":
                    print(f"  - Stored requirement: {result}")
                elif tool_name == "save_files":
                    if isinstance(result, dict) and result.get("success"):
                        print(f"  - {result['message']}")
                        if result.get("files"):
                            print("    Files:")
                            for f in result["files"]: print(f"      - {f}")
                        # Remind user about the generated website
                        print("\nYour generated website is running at http://127.0.0.1:8080/generated")
                    else:
                        print(f"  - Save command result: {result.get('message', 'Failed to save')}")
                elif tool_name == "search_similar":
                     print(f"  - Searched for '{tool_call.get('query','')}': Found {len(result)} results") # Be more informative
        
        # Show usage if available
        if response.get("usage"):
            usage = response["usage"]
            print(f"\nUsage: Prompt Tokens={usage.get('prompt_tokens', 'N/A')}, Completion Tokens={usage.get('completion_tokens', 'N/A')}, Total Tokens={usage.get('total_tokens', 'N/A')}")

# We're no longer using the blueprint registration approach since we're registering 
# routes directly with app.add_url_rule, which works even after the first request
print("Using direct route registration approach (no blueprint needed)")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "web":
        # Initialize context and register the dynamic routes if any components exist
        context = initialize_agent_context()
        components = context.get_all_components()
        if components:
            print(f"Found {len(components)} existing components. Registering dynamic routes...")
            register_generated_routes(context)
        
        print(f"Starting web interface at http://127.0.0.1:8080 for model {MODEL_NAME}")
        if components:
            print(f"Generated website is available at http://127.0.0.1:8080/generated")
        app.run(host='0.0.0.0', port=8080, debug=False) # Use debug=False for production/stable runs
    else:
        # Use integrated server by default (combines CLI and web server)
        run_integrated_server()