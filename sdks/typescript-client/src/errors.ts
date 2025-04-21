/**
 * Error classes for the EngramDB TypeScript client.
 */

/**
 * Base error class for EngramDB client errors.
 */
export class EngramClientError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'EngramClientError';
    Object.setPrototypeOf(this, EngramClientError.prototype);
  }
}

/**
 * Error thrown when the API returns an error response.
 */
export class ApiError extends EngramClientError {
  public readonly statusCode: number;
  public readonly code?: string;
  public readonly details?: Record<string, any>;

  constructor(message: string, statusCode: number, code?: string, details?: Record<string, any>) {
    super(`API Error (${statusCode}): ${message}`);
    this.name = 'ApiError';
    this.statusCode = statusCode;
    this.code = code;
    this.details = details;
    Object.setPrototypeOf(this, ApiError.prototype);
  }
}

/**
 * Error thrown when authentication fails.
 */
export class AuthenticationError extends ApiError {
  constructor(message: string = 'Authentication failed') {
    super(message, 401);
    this.name = 'AuthenticationError';
    Object.setPrototypeOf(this, AuthenticationError.prototype);
  }
}

/**
 * Error thrown when a connection to the API server cannot be established.
 */
export class ConnectionError extends EngramClientError {
  constructor(message: string = 'Failed to connect to the API server') {
    super(message);
    this.name = 'ConnectionError';
    Object.setPrototypeOf(this, ConnectionError.prototype);
  }
}

/**
 * Error thrown when an API request times out.
 */
export class TimeoutError extends EngramClientError {
  constructor(message: string = 'API request timed out') {
    super(message);
    this.name = 'TimeoutError';
    Object.setPrototypeOf(this, TimeoutError.prototype);
  }
}

/**
 * Error thrown when input validation fails.
 */
export class ValidationError extends EngramClientError {
  public readonly field?: string;

  constructor(message: string, field?: string) {
    super(`Validation error${field ? ` for field ${field}` : ''}: ${message}`);
    this.name = 'ValidationError';
    this.field = field;
    Object.setPrototypeOf(this, ValidationError.prototype);
  }
}