"""
Exception classes for the EngramDB client library.
"""

from typing import Any, Dict, Optional, Union


class EngramClientError(Exception):
    """Base exception class for EngramDB client errors."""
    
    def __init__(self, message: str):
        self.message = message
        super().__init__(message)


class ApiError(EngramClientError):
    """Exception raised when the API returns an error response."""
    
    def __init__(
        self, 
        message: str, 
        status_code: int, 
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.code = code
        self.details = details or {}
        super().__init__(f"API Error ({status_code}): {message}")


class AuthenticationError(ApiError):
    """Exception raised when authentication fails."""
    
    def __init__(self, message: str = "Authentication failed"):
        super().__init__(message, 401)


class ConnectionError(EngramClientError):
    """Exception raised when a connection to the API server cannot be established."""
    
    def __init__(self, message: str = "Failed to connect to the API server"):
        super().__init__(message)


class TimeoutError(EngramClientError):
    """Exception raised when an API request times out."""
    
    def __init__(self, message: str = "API request timed out"):
        super().__init__(message)


class ValidationError(EngramClientError):
    """Exception raised when input validation fails."""
    
    def __init__(self, message: str, field: Optional[str] = None):
        self.field = field
        super().__init__(
            f"Validation error{f' for field {field}' if field else ''}: {message}"
        )