"""
Final test script to verify our fixes work correctly

This script:
1. Creates a simple Flask app
2. Adds some test components
3. Saves them to /tmp/generated_flask_website
4. Serves the generated website at http://localhost:8080/generated

Run this with:
python temp_part1.py
"""

import os
import json
import uuid
import re
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np
from flask import Flask, request, jsonify, render_template_string, Response, send_from_directory

# Create test application
app = Flask(__name__)
OUTPUT_PATH = "/tmp/generated_flask_website"

# Create test components
def create_test_components():
    """Create test components for our demo"""
    output_path = Path(OUTPUT_PATH)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "templates").mkdir(exist_ok=True)
    (output_path / "static").mkdir(exist_ok=True)
    (output_path / "static" / "css").mkdir(exist_ok=True)
    
    # Create app.py
    app_code = """from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('index.html', page='about')

if __name__ == '__main__':
    app.run(debug=True)
"""
    # Create index.html
    index_code = """<!DOCTYPE html>
<html>
<head>
    <title>Generated Test App</title>
    <link rel="stylesheet" href="/static/css/style.css">
</head>
<body>
    <div class="container">
        <h1>Generated Test App</h1>
        <p>This is a demonstration of the generated website serving correctly.</p>
        
        {% if page == 'about' %}
        <h2>About Page</h2>
        <p>This is the about page content.</p>
        {% else %}
        <h2>Home Page</h2>
        <p>This is the home page content.</p>
        {% endif %}
        
        <div class="navigation">
            <a href="/generated/">Home</a> | 
            <a href="/generated/about">About</a> | 
            <a href="/">Return to Generator</a>
        </div>
    </div>
</body>
</html>
"""
    # Create style.css
    style_code = """body {
    font-family: Arial, sans-serif;
    margin: 0;
    padding: 0;
    background-color: #f4f4f4;
}

.container {
    max-width: 800px;
    margin: 40px auto;
    padding: 20px;
    background-color: white;
    box-shadow: 0 0 10px rgba(0, 0, 0, 0.1);
    border-radius: 5px;
}

h1 {
    color: #2c3e50;
    border-bottom: 1px solid #eee;
    padding-bottom: 10px;
}

h2 {
    color: #3498db;
}

.navigation {
    margin-top: 30px;
    padding-top: 20px;
    border-top: 1px solid #eee;
}

a {
    color: #3498db;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}
"""
    
    # Write files
    with open(output_path / "app.py", 'w') as f:
        f.write(app_code)
    
    with open(output_path / "templates" / "index.html", 'w') as f:
        f.write(index_code)
        
    with open(output_path / "static" / "css" / "style.css", 'w') as f:
        f.write(style_code)
        
    print(f"Created test components in {output_path}")
    
# Routes
@app.route('/')
def home():
    """Home page - shows information about the generator"""
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
        <p>This is a simplified test version of the Flask Website Generator.</p>
        
        <p>We've created some test components for you to demonstrate the fix:</p>
        <ul>
            <li>app.py - A simple Flask application</li>
            <li>index.html - The main template</li>
            <li>style.css - CSS styling</li>
        </ul>
        
        <p><a href="/generated" class="btn">View Generated Website</a></p>
    </body>
    </html>
    ''')

@app.route('/generated', endpoint='generated_site.home')
@app.route('/generated/', endpoint='generated_site.home')
def generated_site_home():
    """Serves the generated website"""
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
    return send_from_directory(output_path / "static", filename)

# Add main function to run the test server
if __name__ == "__main__":
    # Create test components
    create_test_components()
    
    # Run the Flask server
    print("Starting Flask server on http://localhost:8080")
    print("Visit http://localhost:8080/generated to see the generated website")
    app.run(host='0.0.0.0', port=8080, debug=False)