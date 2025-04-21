//! Configuration for the EngramDB API server

use once_cell::sync::Lazy;
use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use uuid::Uuid;
use crate::database::Database;

/// Global state for the API server
pub struct ApiState {
    /// Active database connections mapped by ID
    pub databases: RwLock<HashMap<String, Arc<RwLock<Database>>>>,
}

impl ApiState {
    /// Create a new ApiState instance
    pub fn new() -> Self {
        Self {
            databases: RwLock::new(HashMap::new()),
        }
    }

    /// Get a database by ID
    pub fn get_database(&self, db_id: &str) -> Option<Arc<RwLock<Database>>> {
        self.databases.read().unwrap().get(db_id).cloned()
    }

    /// Add a database
    pub fn add_database(&self, db_id: String, db: Database) -> String {
        let mut databases = self.databases.write().unwrap();
        let db_arc = Arc::new(RwLock::new(db));
        databases.insert(db_id.clone(), db_arc);
        db_id
    }

    /// Remove a database
    pub fn remove_database(&self, db_id: &str) -> bool {
        let mut databases = self.databases.write().unwrap();
        databases.remove(db_id).is_some()
    }

    /// List all database IDs
    pub fn list_databases(&self) -> Vec<String> {
        let databases = self.databases.read().unwrap();
        databases.keys().cloned().collect()
    }
}

/// Global API state instance
pub static API_STATE: Lazy<ApiState> = Lazy::new(ApiState::new);

/// Configuration for the API server
#[derive(Debug, Clone)]
pub struct ApiConfig {
    /// Port to listen on
    pub port: u16,
    /// Host to bind to
    pub host: String,
    /// Secret key for JWT authentication
    pub jwt_secret: String,
    /// Whether to enable Swagger UI documentation
    pub enable_swagger: bool,
}

impl Default for ApiConfig {
    fn default() -> Self {
        Self {
            port: 8000,
            host: "0.0.0.0".to_string(),
            jwt_secret: Uuid::new_v4().to_string(),
            enable_swagger: true,
        }
    }
}

impl ApiConfig {
    /// Create a new ApiConfig from environment variables
    pub fn from_env() -> Self {
        dotenv::dotenv().ok();
        
        Self {
            port: std::env::var("ENGRAMDB_API_PORT")
                .unwrap_or_else(|_| "8000".to_string())
                .parse()
                .unwrap_or(8000),
            host: std::env::var("ENGRAMDB_API_HOST")
                .unwrap_or_else(|_| "0.0.0.0".to_string()),
            jwt_secret: std::env::var("ENGRAMDB_JWT_SECRET")
                .unwrap_or_else(|_| Uuid::new_v4().to_string()),
            enable_swagger: std::env::var("ENGRAMDB_ENABLE_SWAGGER")
                .unwrap_or_else(|_| "true".to_string())
                .parse()
                .unwrap_or(true),
        }
    }
}