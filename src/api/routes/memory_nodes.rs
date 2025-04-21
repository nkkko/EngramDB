//! Memory node routes

use crate::api::auth::{ApiKey, User};
use crate::api::config::API_STATE;
use crate::api::error::{ApiError, ApiResult};
use crate::api::models::memory_node_models::*;
use crate::core::memory_node::MemoryNode;
use chrono::Utc;
use rocket::http::Status;
use rocket::serde::json::Json;
use uuid::Uuid;

/// List all memory nodes
#[rocket::get("/?<limit>&<offset>&<include_vectors>&<include_connections>")]
pub fn list_nodes(
    _api_key: ApiKey,
    database_id: String,
    limit: Option<usize>,
    offset: Option<usize>,
    include_vectors: Option<bool>,
    include_connections: Option<bool>,
) -> ApiResult<Json<Vec<MemoryNodeOutput>>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let db = db_arc.read().unwrap();

    // Get all memory nodes
    let node_ids = db
        .list_all()
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Apply offset and limit
    let offset = offset.unwrap_or(0);
    let limit = limit.unwrap_or(100);

    let node_ids = node_ids
        .into_iter()
        .skip(offset)
        .take(limit)
        .collect::<Vec<_>>();

    // Load each memory node
    let mut nodes = Vec::with_capacity(node_ids.len());

    for node_id in node_ids {
        let node = db
            .load(node_id)
            .map_err(|e| ApiError::from_engramdb_error(e))?;

        // Convert to output model
        let output = memory_node_to_output(
            node,
            include_vectors.unwrap_or(false),
            include_connections.unwrap_or(false),
        );

        nodes.push(output);
    }

    Ok(Json(nodes))
}

/// Create a new memory node
#[rocket::post("/", data = "<input>")]
pub fn create_node(
    _api_key: ApiKey,
    database_id: String,
    input: Json<MemoryNodeCreateInput>,
) -> ApiResult<Json<MemoryNodeOutput>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let mut db = db_arc.write().unwrap();

    // Create a new memory node
    let mut node = MemoryNode::new(input.vector.clone());

    // Set attributes
    for (key, value) in &input.attributes {
        node.set_attribute(key, value.clone());
    }

    // Set connections (if any)
    for conn in &input.connections {
        // In a real implementation, we'd validate that the target exists
        node.add_connection(conn.target_id, &conn.type_name, conn.strength);
    }

    // Set content (if any)
    if let Some(content) = &input.content {
        node.set_content(content);
    }

    // Save the node to the database
    let node_id = db
        .save(&node)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Reload the node to get the updated version
    let node = db
        .load(node_id)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Convert to output model
    let output = memory_node_to_output(node, true, true);

    Ok(Json(output))
}

/// Create a memory node from text content
#[rocket::post("/from_content", data = "<input>")]
pub fn create_node_from_content(
    _api_key: ApiKey,
    database_id: String,
    input: Json<MemoryNodeContentCreateInput>,
) -> ApiResult<Json<MemoryNodeOutput>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let mut db = db_arc.write().unwrap();

    // Create a new memory node from content
    // In a real implementation, we'd use the specified model
    let mut node = MemoryNode::from_text(&input.content, None)
        .map_err(|_| ApiError::bad_request("Failed to create embedding from content"))?;

    // Set attributes
    for (key, value) in &input.attributes {
        node.set_attribute(key, value.clone());
    }

    // Save the node to the database
    let node_id = db
        .save(&node)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Reload the node to get the updated version
    let node = db
        .load(node_id)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Convert to output model
    let output = memory_node_to_output(node, true, true);

    Ok(Json(output))
}

/// Get a memory node
#[rocket::get("/<node_id>?<include_vectors>&<include_connections>")]
pub fn get_node(
    _api_key: ApiKey,
    database_id: String,
    node_id: String,
    include_vectors: Option<bool>,
    include_connections: Option<bool>,
) -> ApiResult<Json<MemoryNodeOutput>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let db = db_arc.read().unwrap();

    // Parse the node ID
    let node_id =
        Uuid::parse_str(&node_id).map_err(|_| ApiError::bad_request("Invalid node ID format"))?;

    // Load the node
    let node = db
        .load(node_id)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Convert to output model
    let output = memory_node_to_output(
        node,
        include_vectors.unwrap_or(false),
        include_connections.unwrap_or(false),
    );

    Ok(Json(output))
}

/// Update a memory node
#[rocket::put("/<node_id>", data = "<input>")]
pub fn update_node(
    _api_key: ApiKey,
    database_id: String,
    node_id: String,
    input: Json<MemoryNodeUpdateInput>,
) -> ApiResult<Json<MemoryNodeOutput>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let mut db = db_arc.write().unwrap();

    // Parse the node ID
    let node_id =
        Uuid::parse_str(&node_id).map_err(|_| ApiError::bad_request("Invalid node ID format"))?;

    // Load the existing node
    let mut node = db
        .load(node_id)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Update vector if provided
    if let Some(vector) = &input.vector {
        node = MemoryNode::with_id(node_id, vector.clone());

        // Restore attributes and connections
        // This is a simplification; in a real implementation we'd
        // have a way to update the vector directly
    }

    // Update attributes if provided
    if let Some(attributes) = &input.attributes {
        // Clear existing attributes and set new ones
        // In a real implementation, we might merge instead
        node.clear_attributes();

        for (key, value) in attributes {
            node.set_attribute(key, value.clone());
        }
    }

    // Update connections if provided
    if let Some(connections) = &input.connections {
        // Clear existing connections and set new ones
        // In a real implementation, we might have a more granular approach
        node.clear_connections();

        for conn in connections {
            node.add_connection(conn.target_id, &conn.type_name, conn.strength);
        }
    }

    // Update content if provided
    if let Some(content) = &input.content {
        node.set_content(content);
    }

    // Save the updated node
    db.save(&node)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Reload to get the final version
    let node = db
        .load(node_id)
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Convert to output model
    let output = memory_node_to_output(node, true, true);

    Ok(Json(output))
}

/// Delete a memory node
#[rocket::delete("/<node_id>")]
pub fn delete_node(_api_key: ApiKey, database_id: String, node_id: String) -> Status {
    // Get the database from the API state
    let db_arc = match API_STATE.get_database(&database_id) {
        Some(db) => db,
        None => return Status::NotFound,
    };

    let mut db = db_arc.write().unwrap();

    // Parse the node ID
    let node_id = match Uuid::parse_str(&node_id) {
        Ok(id) => id,
        Err(_) => return Status::BadRequest,
    };

    // Delete the node
    match db.delete(node_id) {
        Ok(_) => Status::NoContent,
        Err(_) => Status::NotFound,
    }
}

// Helper function to convert a MemoryNode to a MemoryNodeOutput
fn memory_node_to_output(
    node: MemoryNode,
    include_vectors: bool,
    include_connections: bool,
) -> MemoryNodeOutput {
    // Convert connections if requested
    let connections = if include_connections {
        Some(
            node.connections()
                .into_iter()
                .map(|conn| ConnectionOutput {
                    source_id: node.id(),
                    target_id: conn.target_id,
                    type_name: conn.type_name,
                    strength: conn.strength,
                    custom_type: None, // We don't store custom types currently
                })
                .collect(),
        )
    } else {
        None
    };

    MemoryNodeOutput {
        id: node.id(),
        vector: if include_vectors {
            Some(node.vector().to_vec())
        } else {
            None
        },
        attributes: node.attributes().clone(),
        connections,
        created_at: node.created_at(),
        updated_at: node.updated_at(),
        content: node.content().map(|s| s.to_string()),
    }
}
