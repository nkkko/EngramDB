//! Authentication and authorization for the EngramDB API server

use crate::api::config;
use crate::api::error::{ApiError, ApiResult};
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use rocket::http::Status;
use rocket::request::{FromRequest, Outcome, Request};
use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Claims for JWT authentication
#[derive(Debug, Serialize, Deserialize)]
pub struct Claims {
    /// Subject (user ID)
    pub sub: String,
    /// Expiration time
    pub exp: u64,
    /// Issued at
    pub iat: u64,
}

/// Generate a JWT token for a user
pub fn generate_token(user_id: &str) -> ApiResult<String> {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|_| ApiError::internal_error())?
        .as_secs();
    
    let claims = Claims {
        sub: user_id.to_string(),
        exp: now + 60 * 60 * 24, // 24 hours
        iat: now,
    };
    
    encode(
        &Header::default(),
        &claims,
        &EncodingKey::from_secret(config::API_STATE.jwt_secret.as_bytes()),
    )
    .map_err(|_| ApiError::internal_error())
}

/// Validate a JWT token
pub fn validate_token(token: &str) -> ApiResult<Claims> {
    let token_data = decode::<Claims>(
        token,
        &DecodingKey::from_secret(config::API_STATE.jwt_secret.as_bytes()),
        &Validation::default(),
    )
    .map_err(|_| ApiError::unauthorized())?;
    
    Ok(token_data.claims)
}

/// User with authentication information
pub struct User {
    pub id: String,
}

#[rocket::async_trait]
impl<'r> FromRequest<'r> for User {
    type Error = ApiError;

    async fn from_request(req: &'r Request<'_>) -> Outcome<Self, Self::Error> {
        // Extract token from Authorization header
        let token = match req.headers().get_one("Authorization") {
            Some(token) if token.starts_with("Bearer ") => token[7..].to_string(),
            _ => return Outcome::Failure((Status::Unauthorized, ApiError::unauthorized())),
        };
        
        // Validate token
        match validate_token(&token) {
            Ok(claims) => Outcome::Success(User { id: claims.sub }),
            Err(err) => Outcome::Failure((Status::Unauthorized, err)),
        }
    }
}

/// API key authentication
pub struct ApiKey {
    pub key: String,
}

#[rocket::async_trait]
impl<'r> FromRequest<'r> for ApiKey {
    type Error = ApiError;

    async fn from_request(req: &'r Request<'_>) -> Outcome<Self, Self::Error> {
        // Extract API key from X-API-Key header
        match req.headers().get_one("X-API-Key") {
            Some(key) => {
                // In a real implementation, validate the API key against a database
                // For now, we accept any key for demonstration purposes
                Outcome::Success(ApiKey { key: key.to_string() })
            }
            None => Outcome::Failure((Status::Unauthorized, ApiError::unauthorized())),
        }
    }
}