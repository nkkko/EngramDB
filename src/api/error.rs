//! Error handling for the EngramDB API server

use rocket::http::Status;
use rocket::request::Request;
use rocket::response::{self, Responder, Response};
use serde::{Deserialize, Serialize};
use std::io::Cursor;

/// API error types
#[derive(Debug, Serialize, Deserialize)]
pub struct ApiError {
    pub code: String,
    pub message: String,
    pub status: u16,
}

impl ApiError {
    /// Create a new ApiError
    pub fn new(code: &str, message: &str, status: u16) -> Self {
        Self {
            code: code.to_string(),
            message: message.to_string(),
            status,
        }
    }

    /// Create a not found error
    pub fn not_found(resource: &str) -> Self {
        Self::new(
            "NOT_FOUND",
            &format!("The requested {} was not found", resource),
            404,
        )
    }

    /// Create a bad request error
    pub fn bad_request(message: &str) -> Self {
        Self::new("BAD_REQUEST", message, 400)
    }

    /// Create an unauthorized error
    pub fn unauthorized() -> Self {
        Self::new(
            "UNAUTHORIZED",
            "You are not authorized to perform this action",
            401,
        )
    }

    /// Create a forbidden error
    pub fn forbidden() -> Self {
        Self::new(
            "FORBIDDEN",
            "You do not have permission to access this resource",
            403,
        )
    }

    /// Create an internal server error
    pub fn internal_error() -> Self {
        Self::new(
            "INTERNAL_SERVER_ERROR",
            "An internal server error occurred",
            500,
        )
    }

    /// Create an error from an EngramDB error
    pub fn from_engramdb_error(err: crate::storage::EngramDbError) -> Self {
        match err {
            crate::storage::EngramDbError::Storage(msg) => Self::new("STORAGE_ERROR", &msg, 500),
            crate::storage::EngramDbError::Query(msg) => Self::new("QUERY_ERROR", &msg, 400),
            crate::storage::EngramDbError::Vector(msg) => Self::new("VECTOR_ERROR", &msg, 400),
            crate::storage::EngramDbError::Serialization(msg) => {
                Self::new("SERIALIZATION_ERROR", &msg, 500)
            }
            crate::storage::EngramDbError::Validation(msg) => {
                Self::new("VALIDATION_ERROR", &msg, 400)
            }
            crate::storage::EngramDbError::Other(msg) => Self::new("ERROR", &msg, 500),
        }
    }
}

impl std::fmt::Display for ApiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.code, self.message)
    }
}

impl std::error::Error for ApiError {}

impl<'r> Responder<'r, 'static> for ApiError {
    fn respond_to(self, _: &'r Request<'_>) -> response::Result<'static> {
        let status = Status::from_code(self.status).unwrap_or(Status::InternalServerError);
        let json = serde_json::to_string(&self).unwrap_or_else(|_| {
            r#"{"code":"SERIALIZATION_ERROR","message":"Failed to serialize error response","status":500}"#.to_string()
        });
        Response::build()
            .header(rocket::http::ContentType::JSON)
            .status(status)
            .sized_body(json.len(), Cursor::new(json))
            .ok()
    }
}

/// Result type alias for API operations
pub type ApiResult<T> = Result<T, ApiError>;
