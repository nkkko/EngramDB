//! Connection management routes

use crate::api::auth::{ApiKey, User};
use crate::api::config::API_STATE;
use crate::api::error::{ApiError, ApiResult};
use crate::api::models::connection_models::*;
use rocket::http::Status;
use rocket::serde::json::Json;
use uuid::Uuid;

/// Get connections for a memory node
#[rocket::get("/")]
pub fn get_node_connections(
    _api_key: ApiKey,
    database_id: String,
    node_id: String,
) -> ApiResult<Json<Vec<ConnectionDetails>>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let db = db_arc.read().unwrap();

    // Parse the node ID
    let node_id =
        Uuid::parse_str(&node_id).map_err(|_| ApiError::bad_request("Invalid node ID format"))?;

    // Get the connections
    let connections = db
        .get_connections(node_id)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Convert to output model
    let connection_details = connections
        .into_iter()
        .map(|conn| ConnectionDetails {
            source_id: node_id,
            target_id: conn.target_id,
            type_name: conn.type_name,
            strength: conn.strength,
            custom_type: None, // We don't store custom types currently
        })
        .collect();

    Ok(Json(connection_details))
}

/// Create a connection between memory nodes
#[rocket::post("/", data = "<input>")]
pub fn create_connection(
    _api_key: ApiKey,
    database_id: String,
    node_id: String,
    input: Json<CreateConnectionInput>,
) -> ApiResult<Json<ConnectionDetails>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let mut db = db_arc.write().unwrap();

    // Parse the node ID
    let source_id = Uuid::parse_str(&node_id)
        .map_err(|_| ApiError::bad_request("Invalid source node ID format"))?;

    // Add the connection
    db.add_connection(source_id, input.target_id, &input.type_name, input.strength)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // If bidirectional, add the reverse connection
    if input.bidirectional {
        db.add_connection(input.target_id, source_id, &input.type_name, input.strength)
            .map_err(|e| ApiError::from_engramdb_error(e))?;
    }

    // Prepare the response
    let connection = ConnectionDetails {
        source_id,
        target_id: input.target_id,
        type_name: input.type_name.clone(),
        strength: input.strength,
        custom_type: input.custom_type.clone(),
    };

    Ok(Json(connection))
}

/// Delete a connection
#[rocket::delete("/<target_id>?<bidirectional>")]
pub fn delete_connection(
    _api_key: ApiKey,
    database_id: String,
    node_id: String,
    target_id: String,
    bidirectional: Option<bool>,
) -> Status {
    // Get the database from the API state
    let db_arc = match API_STATE.get_database(&database_id) {
        Some(db) => db,
        None => return Status::NotFound,
    };

    let mut db = db_arc.write().unwrap();

    // Parse the node IDs
    let source_id = match Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => return Status::BadRequest,
    };

    let target_id = match Uuid::parse_str(&target_id) {
        Ok(id) => id,
        Err(_) => return Status::BadRequest,
    };

    // Remove the connection
    let removed = match db.remove_connection(source_id, target_id) {
        Ok(removed) => removed,
        Err(_) => return Status::InternalServerError,
    };

    // If bidirectional, remove the reverse connection
    if bidirectional.unwrap_or(false) {
        let _ = db.remove_connection(target_id, source_id);
    }

    if removed {
        Status::NoContent
    } else {
        Status::NotFound
    }
}
