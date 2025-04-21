/**
 * Memory node class for the EngramDB TypeScript client.
 */

import { EngramDatabase } from './database';
import { Connection, ConnectionInfo, ConnectOptions, MemoryNodeData } from './models';

/**
 * A memory node in the EngramDB database.
 * 
 * This class represents a memory node with vector embeddings, attributes, and connections.
 */
export class MemoryNode {
  /** Node ID */
  readonly id: string;
  
  /** Vector embedding */
  private _vector?: number[];
  
  /** Attributes */
  attributes: Record<string, any>;
  
  /** Connections */
  private _connections: Connection[];
  
  /** Text content */
  private _content?: string;
  
  /** Creation timestamp */
  readonly createdAt: Date;
  
  /** Update timestamp */
  private _updatedAt: Date;
  
  /** Database this node belongs to */
  private database: EngramDatabase;
  
  /** Flag to track local changes */
  private _modified = false;

  /**
   * Initialize a memory node.
   * 
   * @param database - The database this node belongs to
   * @param data - The memory node data
   */
  constructor(database: EngramDatabase, data: MemoryNodeData) {
    this.database = database;
    this.id = data.id;
    this._vector = data.vector;
    this.attributes = data.attributes || {};
    this._connections = data.connections?.map(conn => ({
      targetId: conn.targetId,
      typeName: conn.typeName,
      strength: conn.strength,
      customType: conn.customType,
    })) || [];
    this._content = data.content;
    this.createdAt = new Date(data.createdAt);
    this._updatedAt = new Date(data.updatedAt);
  }

  /**
   * Get the vector embedding for this node.
   * 
   * @returns The vector embedding, or undefined if not available
   */
  get vector(): number[] | undefined {
    return this._vector;
  }

  /**
   * Set the vector embedding for this node.
   * 
   * @param vector - The new vector embedding
   */
  set vector(vector: number[] | undefined) {
    this._vector = vector;
    this._modified = true;
  }

  /**
   * Get the text content associated with this node.
   * 
   * @returns The text content, or undefined if not available
   */
  get content(): string | undefined {
    return this._content;
  }

  /**
   * Set the text content associated with this node.
   * 
   * @param content - The new text content
   */
  set content(content: string | undefined) {
    this._content = content;
    this._modified = true;
  }

  /**
   * Get the last update timestamp.
   * 
   * @returns The last update timestamp
   */
  get updatedAt(): Date {
    return this._updatedAt;
  }

  /**
   * Get all connections for this node.
   * 
   * @returns A list of connections
   */
  get connections(): Connection[] {
    return [...this._connections];
  }

  /**
   * Save changes to this node to the database.
   * 
   * @returns The updated node
   */
  async save(): Promise<MemoryNode> {
    if (!this._modified) {
      return this;
    }
    
    const data: Partial<MemoryNodeData> = {
      attributes: this.attributes,
    };
    
    if (this._vector !== undefined) {
      data.vector = this._vector;
    }
    
    if (this._connections.length > 0) {
      data.connections = this._connections.map(conn => ({
        sourceId: this.id,
        ...conn,
      }));
    }
    
    if (this._content !== undefined) {
      data.content = this._content;
    }
    
    const response = await (this.database as any).client.request<MemoryNodeData>(
      'PUT',
      `databases/${this.database.id}/nodes/${this.id}`,
      undefined,
      data,
    );
    
    // Update local state with server response
    if (response.vector !== undefined) {
      this._vector = response.vector;
    }
    
    this.attributes = response.attributes || {};
    
    if (response.connections) {
      this._connections = response.connections.map(conn => ({
        targetId: conn.targetId,
        typeName: conn.typeName,
        strength: conn.strength,
        customType: conn.customType,
      }));
    }
    
    if (response.content !== undefined) {
      this._content = response.content;
    }
    
    if (response.updatedAt) {
      this._updatedAt = new Date(response.updatedAt);
    }
    
    this._modified = false;
    
    return this;
  }

  /**
   * Refresh the node from the database.
   * 
   * @returns The refreshed node
   */
  async refresh(): Promise<MemoryNode> {
    const node = await this.database.getNode(
      this.id,
      true, // include vectors
      true, // include connections
    );
    
    this._vector = node.vector;
    this.attributes = { ...node.attributes };
    this._connections = [...node.connections];
    this._content = node.content;
    this._updatedAt = node.updatedAt;
    this._modified = false;
    
    return this;
  }

  /**
   * Delete this node from the database.
   * 
   * @returns True if the node was deleted successfully
   */
  async delete(): Promise<boolean> {
    return this.database.deleteNode(this.id);
  }

  /**
   * Get connections from this node.
   * 
   * This method gets connections from the API, not just locally cached ones.
   * 
   * @param relationshipType - Optional relationship type to filter by
   * @returns A list of connections
   */
  async getConnections(relationshipType?: string): Promise<ConnectionInfo[]> {
    const params: Record<string, any> = {};
    
    if (relationshipType) {
      params.relationship_type = relationshipType;
    }
    
    return await (this.database as any).client.request<ConnectionInfo[]>(
      'GET',
      `databases/${this.database.id}/nodes/${this.id}/connections`,
      params,
    );
  }

  /**
   * Create a connection to another node.
   * 
   * This method creates the connection directly via the API.
   * 
   * @param targetId - The ID of the target node
   * @param relationshipType - The type of relationship
   * @param options - Optional connection options
   * @returns The created connection
   */
  async connect(
    targetId: string,
    relationshipType: string,
    options: ConnectOptions = {},
  ): Promise<ConnectionInfo> {
    const data = {
      target_id: targetId,
      type_name: relationshipType,
      strength: options.strength ?? 1.0,
      bidirectional: options.bidirectional ?? false,
    };
    
    if (options.customType) {
      data.custom_type = options.customType;
    }
    
    const response = await (this.database as any).client.request<ConnectionInfo>(
      'POST',
      `databases/${this.database.id}/nodes/${this.id}/connections`,
      undefined,
      data,
    );
    
    // Also update the local connections list
    const connection: Connection = {
      targetId,
      typeName: relationshipType,
      strength: options.strength ?? 1.0,
      customType: options.customType,
    };
    
    this._connections.push(connection);
    
    return response;
  }

  /**
   * Remove a connection to another node.
   * 
   * This method removes the connection directly via the API.
   * 
   * @param targetId - The ID of the target node
   * @param bidirectional - Whether to remove the connection in both directions
   * @returns True if the connection was removed
   */
  async disconnect(targetId: string, bidirectional: boolean = false): Promise<boolean> {
    const params: Record<string, any> = {};
    
    if (bidirectional) {
      params.bidirectional = true;
    }
    
    await (this.database as any).client.request(
      'DELETE',
      `databases/${this.database.id}/nodes/${this.id}/connections/${targetId}`,
      params,
    );
    
    // Also update the local connections list
    const initialLength = this._connections.length;
    this._connections = this._connections.filter(conn => conn.targetId !== targetId);
    
    return this._connections.length < initialLength;
  }

  /**
   * Add a connection locally.
   * 
   * This method only adds the connection locally. Call save() to persist the changes.
   * 
   * @param targetId - The ID of the target node
   * @param relationshipType - The type of relationship
   * @param strength - The strength of the connection (0.0 to 1.0)
   * @param customType - Optional custom relationship type name
   */
  addConnection(
    targetId: string,
    relationshipType: string,
    strength: number = 1.0,
    customType?: string,
  ): void {
    const connection: Connection = {
      targetId,
      typeName: relationshipType,
      strength,
      customType,
    };
    
    this._connections.push(connection);
    this._modified = true;
  }

  /**
   * Remove a connection locally.
   * 
   * This method only removes the connection locally. Call save() to persist the changes.
   * 
   * @param targetId - The ID of the target node
   * @returns True if the connection was removed, false if it did not exist
   */
  removeConnection(targetId: string): boolean {
    const initialLength = this._connections.length;
    this._connections = this._connections.filter(conn => conn.targetId !== targetId);
    
    const removed = this._connections.length < initialLength;
    if (removed) {
      this._modified = true;
    }
    
    return removed;
  }
}