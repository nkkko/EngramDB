//! Database management routes

use crate::api::auth::{ApiKey, User};
use crate::api::config::API_STATE;
use crate::api::error::{ApiError, ApiResult};
use crate::api::models::database_models::*;
use crate::database::{Database, DatabaseConfig};
use crate::storage::StorageType;
use crate::vector::{VectorAlgorithm, VectorIndexConfig};
use chrono::Utc;
use rocket::http::Status;
use rocket::serde::json::Json;

/// List all databases
#[rocket::get("/")]
pub fn list_databases(_api_key: ApiKey) -> ApiResult<Json<Vec<DatabaseInfo>>> {
    let db_ids = API_STATE.list_databases();
    let mut database_infos = Vec::with_capacity(db_ids.len());

    for db_id in db_ids {
        if let Some(db_arc) = API_STATE.get_database(&db_id) {
            let db = db_arc.read().unwrap();

            // Create a simple info object for now, ideally we'd store more metadata
            let info = DatabaseInfo {
                id: db_id.clone(),
                name: format!("database-{}", db_id),
                storage_type: format!("{:?}", db.storage_type()),
                node_count: db.list_all().map(|ids| ids.len()).unwrap_or(0),
                created_at: Utc::now(),
                config: DatabaseConfigInput::default(), // We should store the original config
            };

            database_infos.push(info);
        }
    }

    Ok(Json(database_infos))
}

/// Create a new database
#[rocket::post("/", data = "<input>")]
pub fn create_database(
    _api_key: ApiKey,
    input: Json<CreateDatabaseInput>,
) -> ApiResult<Json<DatabaseInfo>> {
    let input = input.into_inner();

    // Convert input config to DatabaseConfig
    let storage_type = parse_storage_type(&input.config.storage_type);
    let vector_algorithm = parse_vector_algorithm(&input.config.vector_algorithm);

    let db_config = DatabaseConfig {
        storage_type,
        storage_path: input.config.storage_path.clone(),
        cache_size: input.config.cache_size,
        vector_index_config: VectorIndexConfig {
            algorithm: vector_algorithm,
            hnsw: input.config.hnsw_config.map(|config| config.into()),
        },
    };

    // Create the database
    let db = Database::new(db_config.clone()).map_err(|e| ApiError::from_engramdb_error(e))?;

    // Initialize the database
    // db.initialize().map_err(|e| ApiError::from_engramdb_error(e))?;

    // Generate a database ID
    let db_id = uuid::Uuid::new_v4().to_string();

    // Store the database in the API state
    API_STATE.add_database(db_id.clone(), db);

    // Create the response
    let info = DatabaseInfo {
        id: db_id,
        name: input.name,
        storage_type: input.config.storage_type,
        node_count: 0,
        created_at: Utc::now(),
        config: input.config,
    };

    Ok(Json(info))
}

/// Get database information
#[rocket::get("/<database_id>")]
pub fn get_database(_api_key: ApiKey, database_id: String) -> ApiResult<Json<DatabaseInfo>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let db = db_arc.read().unwrap();

    // Create the response
    let info = DatabaseInfo {
        id: database_id,
        name: "database".to_string(), // Ideally we'd store the name
        storage_type: format!("{:?}", db.storage_type()),
        node_count: db.list_all().map(|ids| ids.len()).unwrap_or(0),
        created_at: Utc::now(), // Ideally we'd store the creation time
        config: DatabaseConfigInput::default(), // We should store the original config
    };

    Ok(Json(info))
}

/// Delete a database
#[rocket::delete("/<database_id>")]
pub fn delete_database(_api_key: ApiKey, database_id: String) -> Status {
    // Remove the database from the API state
    if API_STATE.remove_database(&database_id) {
        Status::NoContent
    } else {
        Status::NotFound
    }
}
