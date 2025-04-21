# EngramDB Python Client

A Python client library for interacting with the EngramDB REST API.

## Installation

```bash
pip install engramdb-client
```

## Basic Usage

```python
from engramdb_client import EngramClient
import numpy as np

# Initialize the client
client = EngramClient(api_url="http://localhost:8000/v1")

# Create a database
db = client.create_database("my_database")

# Create a memory node
vector = np.array([0.1, 0.2, 0.3, 0.4]).astype(np.float32)
node = db.create_node(vector=vector)

# Add attributes to the node
node.set_attribute("title", "Important memory")
node.set_attribute("category", "meeting")
node.save()

# Search for similar nodes
similar_nodes = db.search(vector=vector, limit=5)
for node, similarity in similar_nodes:
    print(f"Node {node.id}: similarity={similarity:.4f}, title={node.get_attribute('title')}")

# Create connections between nodes
node2 = db.create_node(vector=np.array([0.15, 0.25, 0.35, 0.45]).astype(np.float32))
node.connect(node2.id, "Association", strength=0.8)

# Get connections
connections = node.get_connections()
for conn in connections:
    print(f"Connected to {conn['target_id']} with type {conn['type']} (strength: {conn['strength']})")
```

## Create from Content

```python
from engramdb_client import EngramClient

# Initialize the client
client = EngramClient(api_url="http://localhost:8000/v1")
db = client.get_database("my_database")

# Create a node from text content (will use the API to generate embeddings)
node = db.create_node_from_content(
    content="This is an important meeting note about the project timeline.",
    model="default"
)

# Search using text queries
results = db.search_text(
    "project timeline", 
    model="default",
    limit=5
)
```

## Authentication

```python
from engramdb_client import EngramClient

# API Key authentication
client = EngramClient(
    api_url="http://localhost:8000/v1",
    api_key="your-api-key"
)

# JWT authentication
client = EngramClient(
    api_url="http://localhost:8000/v1",
    jwt_token="your-jwt-token"
)
```

## Advanced Usage

For more advanced usage examples, please refer to the [examples](examples/) directory.

## API Compatibility

This client library is designed to be compatible with the EngramDB REST API version 1.0 and higher.
The interface is also designed to closely match the PyO3 bindings API for easy migration between the two.

## Development

```bash
# Clone the repository
git clone https://github.com/nkkko/engramdb.git
cd engramdb/sdks/python-client

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.