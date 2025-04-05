# EngramDB Web Interface

A simple Flask-based web interface for managing the EngramDB database.

## Features

- **Create Memory Nodes**: Add new memories with custom attributes and vector embeddings
- **View Memory Details**: Inspect memory attributes and embedding vectors
- **Edit Memories**: Update existing memories with new attributes or embeddings
- **Delete Memories**: Remove unwanted memories from the database
- **Vector Search**: Find similar memories using vector embeddings
- **Attribute Search**: Filter memories by category and importance

## Installation

1. Install requirements:

```bash
pip install -r requirements.txt
```

2. Make sure the EngramDB Python package is installed:

```bash
cd ..
pip install -e python/
```

## Usage

Start the web server:

```bash
python app.py
```

This will start the Flask development server at http://localhost:5000

## Development

The web interface is built with:

- **Flask**: Web framework
- **Bootstrap 5**: Frontend framework
- **JavaScript**: Interactive features

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
  - `base.html` - Base template with layout and navigation
  - `index.html` - Memory list page
  - `create.html` - Create memory form
  - `view.html` - Memory details view
  - `edit.html` - Edit memory form
  - `search.html` - Search interface
- `static/` - Static assets
  - `css/style.css` - Custom styles

## Notes

- The web interface uses an EngramDB file-based database located in the `engramdb_data` directory
- If creating a file-based database fails, it will fall back to an in-memory database
- The interface supports JSON or space-separated number formats for vector embeddings