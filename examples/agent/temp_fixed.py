"""
Standalone Flask Website Generator (Fixed Version)

This is a complete, standalone version of the Flask Website Generator that doesn't depend on EngramDB.
It fixes all the path handling issues and serves the generated website correctly.

Usage:
  python temp_fixed.py
  
Then visit: http://localhost:8080/generated
"""
import os
import json
import uuid
import re
from pathlib import Path
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, Response, send_from_directory

# Test file path handling with and without extensions
def test_path_handling():
    output_path = "/tmp/generated_flask_website"
    output_base = Path(output_path)
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "templates").mkdir(exist_ok=True)
    (output_base / "static").mkdir(exist_ok=True)
    (output_base / "static" / "css").mkdir(exist_ok=True)
    (output_base / "static" / "js").mkdir(exist_ok=True)
    
    # Test component types and file extension handling
    test_cases = [
        # (component_name, component_type, expected_filepath)
        ("app", "main", output_base / "app.py"),
        ("app.py", "main", output_base / "app.py"),
        ("index", "template", output_base / "templates" / "index.html"),
        ("index.html", "template", output_base / "templates" / "index.html"),
        ("style", "static", output_base / "static" / "css" / "style.css"),  # This should be handled by the css logic
        ("style.css", "static", output_base / "static" / "css" / "style.css"),
        ("script", "static", output_base / "static" / "js" / "script.js"),  # This should be handled by the js logic
        ("script.js", "static", output_base / "static" / "js" / "script.js"),
        ("data", "static", output_base / "static" / "data.txt"),  # Default case
        ("data.json", "static", output_base / "static" / "data.json"),  # Custom extension
    ]
    
    print("\nTesting fixed file path handling:")
    for name, comp_type, expected_path in test_cases:
        filepath = get_component_filepath(name, comp_type, output_base)
        print(f"Component: {name} ({comp_type})")
        print(f"  Expected: {expected_path}")
        print(f"  Actual  : {filepath}")
        print(f"  Match   : {expected_path == filepath}")

def get_component_filepath(name, comp_type, output_base):
    """Get the component filepath with proper extension handling"""
    if comp_type == "route" or comp_type == "main":
        # Don't add .py extension if name already has it
        filename = "app.py" if name == "main" else (name if name.endswith('.py') else f"{name}.py")
        return output_base / filename
    elif comp_type == "template":
        # Don't add .html extension if name already has it
        filename = name if name.endswith('.html') else f"{name}.html"
        return output_base / "templates" / filename
    elif comp_type == "static":
        # Determine the file type and path based on content and name
        
        # For exactly "style" or names containing "css" but not ending with another extension
        if name == "style" or ("css" in name.lower() and not any(name.lower().endswith(ext) for ext in ['.js', '.json', '.txt', '.html'] if ext != '.css')):
            # Don't add .css extension if already present
            filename = name if name.endswith('.css') else f"{name}.css"
            return output_base / "static" / "css" / filename
            
        # For exactly "script" or names containing "js" but not ending with another extension 
        elif name == "script" or (("js" in name.lower() or "script" in name.lower()) and not any(name.lower().endswith(ext) for ext in ['.css', '.json', '.txt', '.html'] if ext != '.js')):
            # Don't add .js extension if already present
            filename = name if name.endswith('.js') else f"{name}.js"
            return output_base / "static" / "js" / filename
            
        # For other files, use their extension if present, otherwise add .txt
        else:
            if '.' in name:
                # Keep the extension as-is
                return output_base / "static" / name
            else:
                # Add .txt for files without extension
                return output_base / "static" / f"{name}.txt"
    elif comp_type == "model":
        return output_base / "models.py"
    elif comp_type == "form":
        return output_base / "forms.py"
    else:
        # Default for unknown types
        if '.' in name:
            # Use the name as-is if it already has an extension
            return output_base / name
        else:
            # Default to .py for python-looking components
            return output_base / f"{name}.py"

# Function to actually write files
def write_test_components():
    """Create and write test components to verify file path handling"""
    output_path = "/tmp/generated_flask_website"
    output_base = Path(output_path)
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "templates").mkdir(exist_ok=True)
    (output_base / "static").mkdir(exist_ok=True)
    (output_base / "static" / "css").mkdir(exist_ok=True)
    (output_base / "static" / "js").mkdir(exist_ok=True)
    
    # Create test components
    components = [
        {
            "id": str(uuid.uuid4()),
            "name": "app.py",
            "type": "main",
            "code": "from flask import Flask, render_template\n\napp = Flask(__name__)\n\n@app.route('/')\ndef home():\n    return render_template('index.html')\n\nif __name__ == '__main__':\n    app.run(debug=True)",
            "description": "The main Flask application",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "index",  # No extension
            "type": "template",
            "code": "<!DOCTYPE html>\n<html>\n<head>\n    <title>Test App</title>\n    <link rel=\"stylesheet\" href=\"/static/css/style.css\">\n</head>\n<body>\n    <h1>Test App</h1>\n</body>\n</html>",
            "description": "The main template",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "style",  # No extension
            "type": "static",
            "code": "body { font-family: Arial; margin: 0; padding: 20px; }\nh1 { color: blue; }",
            "description": "CSS styles",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": str(uuid.uuid4()),
            "name": "data.json",  # With extension
            "type": "static",
            "code": '{"test": true, "value": 123}',
            "description": "Test JSON data",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    print(f"Writing test components to {output_path}...")
    for component in components:
        name = component["name"]
        comp_type = component["type"]
        code = component["code"]
        
        filepath = get_component_filepath(name, comp_type, output_base)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the file
        with open(filepath, 'w') as f:
            f.write(code)
        
        print(f"Wrote component: {name} ({comp_type}) to {filepath}")
    
    # Verify files were written correctly
    print("\nVerifying files...")
    expected_paths = [
        output_base / "app.py",
        output_base / "templates" / "index.html",
        output_base / "static" / "css" / "style.css",
        output_base / "static" / "data.json"
    ]
    
    for path in expected_paths:
        exists = path.exists()
        print(f"- {path}: {'EXISTS' if exists else 'MISSING'}")

# Create the Flask app
app = Flask(__name__)
OUTPUT_PATH = "/tmp/generated_flask_website"

# Routes
@app.route('/')
def home():
    """Home page for the website generator"""
    # First, make sure we have some components
    write_test_components()
    
    return render_template_string('''
    <html>
    <head>
        <title>Flask Website Generator</title>
        <style>
            body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; line-height: 1.6; }
            .btn { display: inline-block; padding: 10px 15px; background: #4CAF50; color: white; text-decoration: none; border-radius: 4px; margin-right: 10px; }
            .btn:hover { background: #45a049; }
        </style>
    </head>
    <body>
        <h1>Flask Website Generator</h1>
        <p>This is a fixed, standalone version of the website generator that doesn't depend on EngramDB.</p>
        
        <div style="margin: 20px 0; padding: 15px; background-color: #e8f5e9; border-left: 5px solid #4CAF50;">
            <h3>Features Fixed:</h3>
            <ul>
                <li>Correct path handling for all file types</li>
                <li>Proper handling of file extensions</li>
                <li>Fixed component organization by type (CSS, JS, etc.)</li>
                <li>Proper serving of the generated website</li>
            </ul>
        </div>
        
        <p>Test components have been created at: <code>{{ output_path }}</code></p>
        <p><a href="/generated" class="btn">View Generated Website</a></p>
        
        <h3>Files Generated:</h3>
        <ul>
            <li>app.py - Main Flask application</li>
            <li>templates/index.html - Website template</li>
            <li>static/css/style.css - CSS styling</li>
        </ul>
    </body>
    </html>
    ''', output_path=OUTPUT_PATH)

@app.route('/generated', endpoint='generated_site.home')
@app.route('/generated/', endpoint='generated_site.home')
def generated_site_home():
    """Serve the generated website"""
    output_path = Path(OUTPUT_PATH)
    index_html_path = output_path / "templates" / "index.html"
    
    if index_html_path.exists():
        print(f"Found index.html at {index_html_path}, serving it")
        
        # Read the template
        with open(index_html_path, 'r') as f:
            template_content = f.read()
            
        # Fix paths for static files
        template_content = template_content.replace('href="/static/', 'href="/generated/static/')
        template_content = template_content.replace('src="/static/', 'src="/generated/static/')
        
        return render_template_string(template_content)
    else:
        return "No generated website found. Please create components first."

@app.route('/generated/<path:subpath>')
def generated_subpaths(subpath):
    """Handle subpaths in the generated website"""
    output_path = Path(OUTPUT_PATH)
    index_html_path = output_path / "templates" / "index.html"
    
    if index_html_path.exists():
        # Read the template
        with open(index_html_path, 'r') as f:
            template_content = f.read()
            
        # Fix paths for static files
        template_content = template_content.replace('href="/static/', 'href="/generated/static/')
        template_content = template_content.replace('src="/static/', 'src="/generated/static/')
        
        # Pass the subpath as a variable to the template
        return render_template_string(template_content, page=subpath)
    else:
        return "No generated website found. Please create components first."

@app.route('/generated/static/<path:filename>')
def generated_static(filename):
    """Serve static files for the generated website"""
    output_path = Path(OUTPUT_PATH)
    try:
        # For CSS files, check in the css subdirectory
        if filename.endswith('.css'):
            if (output_path / "static" / "css" / filename).exists():
                return send_from_directory(output_path / "static" / "css", filename)
            
        # For JS files, check in the js subdirectory
        if filename.endswith('.js'):
            if (output_path / "static" / "js" / filename).exists():
                return send_from_directory(output_path / "static" / "js", filename)
        
        # For all other files, look in the static directory
        return send_from_directory(output_path / "static", filename)
    except Exception as e:
        print(f"Error serving static file: {e}")
        return f"Error: {str(e)}", 404

# Run the app
if __name__ == "__main__":
    # Ensure we have test components
    write_test_components()
    
    # Start the server
    print(f"Starting server on http://localhost:8080")
    print(f"Visit http://localhost:8080/generated to see the generated website")
    app.run(host='0.0.0.0', port=8080)