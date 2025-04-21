/**
 * EngramDB client for connecting to the EngramDB REST API.
 */

import axios, { AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import {
  ClientConfig,
  CreateDatabaseInput,
  DatabaseInfo,
  EmbeddingModelInfo,
  GeneratedEmbedding,
} from './models';
import { EngramDatabase } from './database';
import {
  ApiError,
  AuthenticationError,
  ConnectionError,
  EngramClientError,
  TimeoutError,
} from './errors';

/**
 * Client for interacting with the EngramDB REST API.
 * 
 * This class provides methods for managing databases and connections to the API.
 */
export class EngramClient {
  private apiUrl: string;
  private axios: AxiosInstance;

  /**
   * Initialize the EngramDB client.
   * 
   * @param config - Client configuration options
   */
  constructor(config: ClientConfig) {
    this.apiUrl = config.apiUrl.replace(/\/+$/, ''); // Remove trailing slashes
    
    // Set up headers for authentication
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    
    if (config.apiKey) {
      headers['X-API-Key'] = config.apiKey;
    }
    
    if (config.jwtToken) {
      headers['Authorization'] = `Bearer ${config.jwtToken}`;
    }
    
    // Create axios instance
    this.axios = axios.create({
      baseURL: this.apiUrl,
      headers,
      timeout: config.timeout || 10000,
    });
  }

  /**
   * Make a request to the API.
   * 
   * @param method - The HTTP method
   * @param path - The API endpoint path
   * @param params - Optional query parameters
   * @param data - Optional request body
   * @returns The response data
   * @throws {ApiError} If the API returns an error response
   * @throws {AuthenticationError} If authentication fails
   * @throws {ConnectionError} If a connection cannot be established
   * @throws {TimeoutError} If the request times out
   */
  private async request<T = any>(
    method: string,
    path: string,
    params?: Record<string, any>,
    data?: any,
  ): Promise<T> {
    try {
      const config: AxiosRequestConfig = {
        method,
        url: path.startsWith('/') ? path.substring(1) : path,
        params,
        data,
      };
      
      const response: AxiosResponse<T> = await this.axios.request(config);
      return response.data;
    } catch (error: any) {
      if (error.response) {
        // The request was made and the server responded with an error status code
        const { status, data } = error.response;
        
        if (status === 401) {
          throw new AuthenticationError();
        }
        
        throw new ApiError(
          data.message || 'Unknown API error',
          status,
          data.code,
          data.details,
        );
      } else if (error.request) {
        // The request was made but no response was received
        if (error.code === 'ECONNABORTED') {
          throw new TimeoutError();
        } else {
          throw new ConnectionError(error.message || 'Failed to connect to the API server');
        }
      } else {
        // Something happened in setting up the request
        throw new EngramClientError(error.message || 'Request setup error');
      }
    }
  }

  /**
   * List all databases.
   * 
   * @returns A list of database information objects
   */
  async listDatabases(): Promise<DatabaseInfo[]> {
    const response = await this.request<DatabaseInfo[]>('GET', 'databases');
    return response;
  }

  /**
   * Create a new database.
   * 
   * @param nameOrConfig - Database name or configuration object
   * @returns A database client for the newly created database
   */
  async createDatabase(nameOrConfig: string | CreateDatabaseInput): Promise<EngramDatabase> {
    const config: CreateDatabaseInput = typeof nameOrConfig === 'string'
      ? { name: nameOrConfig }
      : nameOrConfig;
    
    const response = await this.request<DatabaseInfo>('POST', 'databases', undefined, config);
    
    return new EngramDatabase(this, response.id, response.name);
  }

  /**
   * Get a database client by ID.
   * 
   * @param databaseId - The ID of the database
   * @returns A database client for the specified database
   */
  async getDatabase(databaseId: string): Promise<EngramDatabase> {
    const response = await this.request<DatabaseInfo>('GET', `databases/${databaseId}`);
    
    return new EngramDatabase(this, response.id, response.name);
  }

  /**
   * Delete a database.
   * 
   * @param databaseId - The ID of the database to delete
   * @returns True if the database was deleted
   */
  async deleteDatabase(databaseId: string): Promise<boolean> {
    await this.request('DELETE', `databases/${databaseId}`);
    return true;
  }

  /**
   * Get a list of available embedding models.
   * 
   * @returns A list of embedding model information
   */
  async getModels(): Promise<EmbeddingModelInfo[]> {
    return await this.request<EmbeddingModelInfo[]>('GET', 'models');
  }

  /**
   * Generate embeddings for text content.
   * 
   * @param content - The text content to embed
   * @param model - The embedding model to use
   * @returns A vector embedding
   */
  async generateEmbedding(content: string, model: string = 'default'): Promise<number[]> {
    const response = await this.request<GeneratedEmbedding>(
      'POST',
      'generate_embedding',
      undefined,
      { content, model },
    );
    
    return response.vector;
  }
}