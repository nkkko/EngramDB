/**
 * Data models for the EngramDB TypeScript client.
 */

/**
 * Configuration for an EngramDB client.
 */
export interface ClientConfig {
  /** Base URL of the EngramDB API */
  apiUrl: string;
  /** Optional API key for authentication */
  apiKey?: string;
  /** Optional JWT token for authentication */
  jwtToken?: string;
  /** Request timeout in milliseconds (default: 10000) */
  timeout?: number;
}

/**
 * Configuration for an EngramDB database.
 */
export interface DatabaseConfig {
  /** Storage type (Memory, MultiFile, SingleFile) */
  storageType?: string;
  /** Storage path for file-based databases */
  storagePath?: string;
  /** Cache size */
  cacheSize?: number;
  /** Vector algorithm (Linear, HNSW) */
  vectorAlgorithm?: string;
  /** HNSW configuration */
  hnswConfig?: HnswConfig;
}

/**
 * Configuration for the HNSW algorithm.
 */
export interface HnswConfig {
  /** Maximum number of connections per node */
  m?: number;
  /** Size of dynamic candidate list during construction */
  efConstruction?: number;
  /** Size of dynamic candidate list during search */
  ef?: number;
  /** Level multiplier */
  levelMultiplier?: number;
  /** Maximum level */
  maxLevel?: number;
}

/**
 * Information about an EngramDB database.
 */
export interface DatabaseInfo {
  /** Database ID */
  id: string;
  /** Database name */
  name: string;
  /** Storage type */
  storageType: string;
  /** Number of nodes in the database */
  nodeCount: number;
  /** Creation timestamp */
  createdAt: string;
  /** Database configuration */
  config?: DatabaseConfig;
}

/**
 * Input for creating a new database.
 */
export interface CreateDatabaseInput {
  /** Database name */
  name: string;
  /** Database configuration */
  config?: DatabaseConfig;
}

/**
 * A connection between memory nodes.
 */
export interface Connection {
  /** Target node ID */
  targetId: string;
  /** Relationship type */
  typeName: string;
  /** Connection strength (0.0 to 1.0) */
  strength?: number;
  /** Custom type name (for Custom relationship type) */
  customType?: string;
}

/**
 * Detailed information about a connection.
 */
export interface ConnectionInfo extends Connection {
  /** Source node ID */
  sourceId: string;
}

/**
 * Types of relationships between memory nodes.
 */
export enum RelationshipType {
  ASSOCIATION = 'Association',
  CAUSATION = 'Causation',
  SEQUENCE = 'Sequence',
  CONTAINS = 'Contains',
  PART_OF = 'PartOf',
  REFERENCE = 'Reference',
  CUSTOM = 'Custom',
}

/**
 * Operations for attribute filters.
 */
export enum FilterOperation {
  EQUALS = 'equals',
  NOT_EQUALS = 'not_equals',
  GREATER_THAN = 'greater_than',
  LESS_THAN = 'less_than',
  GREATER_OR_EQUAL = 'greater_or_equal',
  LESS_OR_EQUAL = 'less_or_equal',
  CONTAINS = 'contains',
  STARTS_WITH = 'starts_with',
  ENDS_WITH = 'ends_with',
  EXISTS = 'exists',
}

/**
 * A filter for querying memory nodes by attributes.
 */
export interface AttributeFilter {
  /** Attribute field name */
  field: string;
  /** Filter operation */
  operation: FilterOperation;
  /** Filter value */
  value?: any;
}

/**
 * A query for searching memory nodes.
 */
export interface SearchQuery {
  /** Vector embedding to search for */
  vector?: number[];
  /** Text content to search for */
  content?: string;
  /** Embedding model to use */
  model?: string;
  /** Attribute filters */
  filters?: AttributeFilter[];
  /** Maximum number of results */
  limit?: number;
  /** Minimum similarity threshold */
  threshold?: number;
  /** Whether to include vector embeddings in results */
  includeVectors?: boolean;
  /** Whether to include connections in results */
  includeConnections?: boolean;
}

/**
 * Options for creating a memory node.
 */
export interface CreateNodeOptions {
  /** Vector embedding */
  vector?: number[];
  /** Attributes */
  attributes?: Record<string, any>;
  /** Connections */
  connections?: Connection[];
  /** Text content */
  content?: string;
}

/**
 * Options for creating a memory node from content.
 */
export interface CreateNodeFromContentOptions {
  /** Text content */
  content: string;
  /** Embedding model to use */
  model?: string;
  /** Attributes */
  attributes?: Record<string, any>;
}

/**
 * Options for connecting memory nodes.
 */
export interface ConnectOptions {
  /** Strength of the connection (0.0 to 1.0) */
  strength?: number;
  /** Custom type name (for Custom relationship type) */
  customType?: string;
  /** Whether to create a bidirectional connection */
  bidirectional?: boolean;
}

/**
 * Data for a memory node.
 */
export interface MemoryNodeData {
  /** Node ID */
  id: string;
  /** Vector embedding */
  vector?: number[];
  /** Attributes */
  attributes: Record<string, any>;
  /** Connections */
  connections?: ConnectionInfo[];
  /** Creation timestamp */
  createdAt: string;
  /** Update timestamp */
  updatedAt: string;
  /** Text content */
  content?: string;
}

/**
 * A search result containing a memory node and its similarity score.
 */
export interface SearchResult {
  /** Memory node */
  node: MemoryNodeData;
  /** Similarity score */
  similarity: number;
}

/**
 * Information about an embedding model.
 */
export interface EmbeddingModelInfo {
  /** Model ID */
  id: string;
  /** Model name */
  name: string;
  /** Vector dimensions */
  dimensions: number;
  /** Model description */
  description: string;
  /** Provider name */
  provider: string;
  /** Model type (single_vector or multi_vector) */
  modelType: string;
}

/**
 * Generated embedding result.
 */
export interface GeneratedEmbedding {
  /** Vector embedding */
  vector: number[];
  /** Model used */
  model: string;
  /** Vector dimensions */
  dimensions: number;
}