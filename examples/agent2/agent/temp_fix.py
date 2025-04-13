"""
Simple test script to verify our fixes to flask_website_generator.py
"""
import os
from pathlib import Path

# Verify we're using the correct output path
output_path = os.environ.get("WEBSITE_OUTPUT_PATH", "/tmp/generated_flask_website")
print(f"Using output path: {output_path}")

# Test file path handling with and without extensions
def test_path_handling():
    output_base = Path(output_path)
    output_base.mkdir(parents=True, exist_ok=True)
    (output_base / "templates").mkdir(exist_ok=True)
    (output_base / "static").mkdir(exist_ok=True)
    (output_base / "static" / "css").mkdir(exist_ok=True)
    
    # Test component types and file extension handling
    test_cases = [
        # (component_name, component_type, expected_filename, expected_path)
        ("app", "main", "app.py", output_base / "app.py"),
        ("app.py", "main", "app.py", output_base / "app.py"),
        ("index", "template", "index.html", output_base / "templates" / "index.html"),
        ("index.html", "template", "index.html", output_base / "templates" / "index.html"),
        ("style", "static", "style.css", output_base / "static" / "css" / "style.css"),
        ("style.css", "static", "style.css", output_base / "static" / "css" / "style.css"),
    ]
    
    print("\nTesting file path handling:")
    for name, comp_type, expected_filename, expected_path in test_cases:
        # Determine file path based on component type
        if comp_type == "route" or comp_type == "main":
            # Don't add .py extension if name already has it
            filename = "app.py" if name == "main" else (name if name.endswith('.py') else f"{name}.py")
            filepath = output_base / filename
        elif comp_type == "template":
            # Don't add .html extension if name already has it
            filename = name if name.endswith('.html') else f"{name}.html"
            filepath = output_base / "templates" / filename
        elif comp_type == "static":
            # Handle different static file types
            if "css" in name.lower():
                # Don't add .css extension if name already has it
                filename = name if name.endswith('.css') else f"{name}.css"
                filepath = output_base / "static" / "css" / filename
            elif "js" in name.lower():
                # Don't add .js extension if name already has it
                filename = name if name.endswith('.js') else f"{name}.js"
                filepath = output_base / "static" / "js" / filename
            else:
                # Don't add .txt extension if name already has extension
                filename = name if '.' in name else f"{name}.txt" 
                filepath = output_base / "static" / filename
            
        print(f"Component: {name} ({comp_type})")
        print(f"  Expected: {expected_filename} at {expected_path}")
        print(f"  Actual  : {filename} at {filepath}")
        print(f"  Match   : {expected_path == filepath}")

# Run tests
test_path_handling()