/**
 * EngramDB database client for interacting with a specific database.
 */

import { EngramClient } from './client';
import { MemoryNode } from './memoryNode';
import {
  AttributeFilter,
  CreateNodeFromContentOptions,
  CreateNodeOptions,
  MemoryNodeData,
  SearchQuery,
  SearchResult,
} from './models';

/**
 * Client for interacting with a specific EngramDB database.
 * 
 * This class provides methods for interacting with memory nodes and performing searches.
 */
export class EngramDatabase {
  /** The database ID */
  readonly id: string;
  
  /** The database name */
  readonly name: string;
  
  /** The EngramDB client to use for API requests */
  private client: EngramClient;

  /**
   * Initialize the database client.
   * 
   * @param client - The EngramDB client to use for API requests
   * @param id - The ID of the database
   * @param name - The name of the database
   */
  constructor(client: EngramClient, id: string, name: string) {
    this.client = client;
    this.id = id;
    this.name = name;
  }

  /**
   * Create a new memory node in the database.
   * 
   * @param options - Options for creating the memory node
   * @returns The created memory node
   */
  async createNode(options: CreateNodeOptions): Promise<MemoryNode> {
    const response = await (this.client as any).request<MemoryNodeData>(
      'POST',
      `databases/${this.id}/nodes`,
      undefined,
      options,
    );
    
    return new MemoryNode(this, response);
  }

  /**
   * Create a new memory node from text content.
   * 
   * This will use the embeddings API to generate an embedding for the content.
   * 
   * @param options - Options for creating the memory node from content
   * @returns The created memory node
   */
  async createNodeFromContent(options: CreateNodeFromContentOptions): Promise<MemoryNode> {
    const response = await (this.client as any).request<MemoryNodeData>(
      'POST',
      `databases/${this.id}/nodes/from_content`,
      undefined,
      options,
    );
    
    return new MemoryNode(this, response);
  }

  /**
   * Get a memory node by ID.
   * 
   * @param nodeId - The ID of the memory node
   * @param includeVectors - Whether to include vector embeddings in the response
   * @param includeConnections - Whether to include connections in the response
   * @returns The memory node
   */
  async getNode(
    nodeId: string,
    includeVectors: boolean = false,
    includeConnections: boolean = false,
  ): Promise<MemoryNode> {
    const params = {
      include_vectors: includeVectors,
      include_connections: includeConnections,
    };
    
    const response = await (this.client as any).request<MemoryNodeData>(
      'GET',
      `databases/${this.id}/nodes/${nodeId}`,
      params,
    );
    
    return new MemoryNode(this, response);
  }

  /**
   * Delete a memory node.
   * 
   * @param nodeId - The ID of the memory node to delete
   * @returns True if the node was deleted
   */
  async deleteNode(nodeId: string): Promise<boolean> {
    await (this.client as any).request(
      'DELETE',
      `databases/${this.id}/nodes/${nodeId}`,
    );
    
    return true;
  }

  /**
   * List memory nodes in the database.
   * 
   * @param limit - Maximum number of nodes to return
   * @param offset - Number of nodes to skip
   * @param includeVectors - Whether to include vector embeddings in the response
   * @param includeConnections - Whether to include connections in the response
   * @returns A list of memory nodes
   */
  async listNodes(
    limit: number = 100,
    offset: number = 0,
    includeVectors: boolean = false,
    includeConnections: boolean = false,
  ): Promise<MemoryNode[]> {
    const params = {
      limit,
      offset,
      include_vectors: includeVectors,
      include_connections: includeConnections,
    };
    
    const response = await (this.client as any).request<{ nodes: MemoryNodeData[] }>(
      'GET',
      `databases/${this.id}/nodes`,
      params,
    );
    
    return response.nodes.map(nodeData => new MemoryNode(this, nodeData));
  }

  /**
   * Search for memory nodes.
   * 
   * @param query - The search query
   * @returns A list of search results
   */
  async search(query: SearchQuery): Promise<{ node: MemoryNode; similarity: number }[]> {
    if (!query.vector && !query.content) {
      throw new Error('Either vector or content must be provided for search');
    }
    
    const response = await (this.client as any).request<{ results: SearchResult[] }>(
      'POST',
      `databases/${this.id}/search`,
      undefined,
      query,
    );
    
    return response.results.map(result => ({
      node: new MemoryNode(this, result.node),
      similarity: result.similarity,
    }));
  }

  /**
   * Search for memory nodes using text.
   * 
   * This is a convenience method that searches using the text content.
   * 
   * @param options - Search options
   * @returns A list of search results
   */
  async searchText(options: {
    text: string;
    model?: string;
    filters?: AttributeFilter[];
    limit?: number;
    threshold?: number;
    includeVectors?: boolean;
    includeConnections?: boolean;
  }): Promise<{ node: MemoryNode; similarity: number }[]> {
    return this.search({
      content: options.text,
      model: options.model || 'default',
      filters: options.filters,
      limit: options.limit,
      threshold: options.threshold,
      includeVectors: options.includeVectors,
      includeConnections: options.includeConnections,
    });
  }

  /**
   * Clear all nodes in the database.
   * 
   * @returns True if successful
   */
  async clearAll(): Promise<boolean> {
    const nodes = await this.listNodes();
    
    // Delete all nodes
    for (const node of nodes) {
      await this.deleteNode(node.id);
    }
    
    return true;
  }
}