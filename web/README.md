# EngramDB Web Interface

A web-based user interface for the EngramDB memory database system.

## Features

- **Create Memory Nodes**: Add new memories with custom attributes and vector embeddings
- **View Memory Details**: Inspect memory attributes and embedding vectors
- **Edit Memories**: Update existing memories with new attributes or embeddings
- **Delete Memories**: Remove unwanted memories from the database
- **Vector Search**: Find similar memories using vector embeddings
- **Attribute Search**: Filter memories by category and importance
- **Visual Representation**: Visual display of vector embeddings
- **AI-Powered Embeddings**: Generate vector embeddings from text using multilingual-e5-large-instruct model
- **Multilingual Support**: Create and search memories in 100+ languages with semantic understanding

## Running the Application

### Quick Start (Recommended)

Run the application using the provided script:

```bash
./run_web.sh
```

This script will:
1. Create a virtual environment (if not exists)
2. Install required dependencies
3. Run the application on port 8082

Once running, access the web interface at: http://localhost:8082

### Manual Setup

If you prefer to set up manually:

1. Create a virtual environment:
   ```bash
   python3 -m venv venv_app
   source venv_app/bin/activate
   ```

2. Install dependencies:
   ```bash
   # Basic dependencies
   pip install flask==2.3.3 werkzeug==2.3.7 flask-wtf==1.2.1
   
   # ML dependencies for embedding model (optional)
   pip install torch transformers sentence-transformers
   ```

3. Run the application:
   ```bash
   python app_full.py
   ```

## Available Applications

- `app_full.py` - Complete application with all features (uses mock implementation if actual EngramDB is not available)
- `app_simple.py` - Simplified version with basic pages only (for demo purposes)
- `app.py` - Original implementation that requires the actual EngramDB library

## Installation Scripts

- `run_web.sh` - Main script to set up and run the web interface
- `install_ml_deps.sh` - Install machine learning dependencies for embedding functionality
- `install_engramdb.sh` - Install the actual EngramDB Python library
- `run_example.sh` - Run the embedding model example

## Project Structure

- `templates/` - HTML templates
  - `base.html` - Base template with layout and navigation
  - `index.html` - Memory list page
  - `create.html` - Create memory form
  - `view.html` - Memory details view
  - `edit.html` - Edit memory form
  - `search.html` - Search interface
- `static/` - Static assets
  - `css/style.css` - Custom styles
- `embedding_utils.py` - Utility functions for text-to-embedding conversion
- `examples/` - Example scripts
  - `embedding_example.py` - Demonstrates embedding generation and search
- `model_cache/` - Cache directory for downloaded embedding models

## Technical Details

- **Backend**: Python Flask web framework
- **Frontend**: Bootstrap 5 with responsive design
- **Database**: Uses EngramDB with file-based storage in the `engramdb_data` directory
- **Mock Implementation**: Will use a mock implementation of EngramDB if the actual library is not available
- **Embedding Formats**: Supports JSON or space-separated number formats for vector embeddings
- **Embedding Model**: Uses multilingual-e5-large-instruct from Hugging Face for text-to-embedding conversion
- **Language Support**: Over 100 languages for embedding generation and semantic search
- **Vector Dimensions**: 1024-dimensional embeddings from the model (or custom dimensions for manual input)