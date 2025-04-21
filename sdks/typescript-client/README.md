# EngramDB TypeScript Client

A TypeScript client library for interacting with the EngramDB REST API.

## Installation

```bash
npm install @engramdb/client
# or
yarn add @engramdb/client
```

## Basic Usage

```typescript
import { EngramClient } from '@engramdb/client';

// Initialize the client
const client = new EngramClient({ apiUrl: 'http://localhost:8000/v1' });

async function example() {
  // Create a database
  const db = await client.createDatabase('my_database');

  // Create a memory node
  const vector = [0.1, 0.2, 0.3, 0.4];
  const node = await db.createNode({ vector });

  // Add attributes to the node
  node.attributes.title = 'Important memory';
  node.attributes.category = 'meeting';
  await node.save();

  // Search for similar nodes
  const searchResults = await db.search({
    vector,
    limit: 5,
  });

  for (const { node, similarity } of searchResults) {
    console.log(`Node ${node.id}: similarity=${similarity.toFixed(4)}, title=${node.attributes.title}`);
  }

  // Create connections between nodes
  const node2 = await db.createNode({
    vector: [0.15, 0.25, 0.35, 0.45],
  });
  
  await node.connect(node2.id, 'Association', { strength: 0.8 });

  // Get connections
  const connections = await node.getConnections();
  for (const conn of connections) {
    console.log(`Connected to ${conn.targetId} with type ${conn.typeName} (strength: ${conn.strength})`);
  }
}
```

## Create from Content

```typescript
import { EngramClient } from '@engramdb/client';

async function contentExample() {
  // Initialize the client
  const client = new EngramClient({ apiUrl: 'http://localhost:8000/v1' });
  const db = await client.getDatabase('my_database');

  // Create a node from text content (will use the API to generate embeddings)
  const node = await db.createNodeFromContent({
    content: 'This is an important meeting note about the project timeline.',
    model: 'default',
  });

  // Search using text queries
  const results = await db.searchText({
    text: 'project timeline',
    model: 'default',
    limit: 5,
  });
}
```

## Authentication

```typescript
import { EngramClient } from '@engramdb/client';

// API Key authentication
const client = new EngramClient({
  apiUrl: 'http://localhost:8000/v1',
  apiKey: 'your-api-key',
});

// JWT authentication
const client = new EngramClient({
  apiUrl: 'http://localhost:8000/v1',
  jwtToken: 'your-jwt-token',
});
```

## Advanced Usage

For more advanced usage examples, please refer to the [examples](examples/) directory.

## API Compatibility

This client library is designed to be compatible with the EngramDB REST API version 1.0 and higher.
The interface is also designed to closely match the EngramDB Python API for easy migration.

## Development

```bash
# Clone the repository
git clone https://github.com/nkkko/engramdb.git
cd engramdb/sdks/typescript-client

# Install dependencies
npm install

# Build the library
npm run build

# Run tests
npm test
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.