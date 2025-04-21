//! Search routes

use crate::api::auth::{ApiKey, User};
use crate::api::config::API_STATE;
use crate::api::error::{ApiError, ApiResult};
use crate::api::models::memory_node_models::*;
use crate::api::models::search_models::*;
use crate::query::AttributeFilter;
use rocket::serde::json::Json;

/// Search for memory nodes
#[rocket::post("/", data = "<input>")]
pub fn search_nodes(
    _api_key: ApiKey,
    database_id: String,
    input: Json<SearchQueryInput>,
) -> ApiResult<Json<SearchResults>> {
    // Get the database from the API state
    let db_arc = API_STATE
        .get_database(&database_id)
        .ok_or_else(|| ApiError::not_found("database"))?;

    let db = db_arc.read().unwrap();

    // Check that we have either a vector or content
    if input.vector.is_none() && input.content.is_none() {
        return Err(ApiError::bad_request(
            "Either vector or content must be provided for search",
        ));
    }

    // If content is provided, convert it to a vector using the specified model
    let query_vector = if let Some(vector) = &input.vector {
        vector.clone()
    } else if let Some(content) = &input.content {
        // Use the embedding service to convert content to vector
        // In a real implementation, we'd use the specified model
        return Err(ApiError::bad_request(
            "Searching by content without a vector is not yet implemented",
        ));
    } else {
        return Err(ApiError::bad_request("No query vector available"));
    };

    // Start building the query
    let mut query_builder = db.query();

    // Add vector similarity search
    query_builder = query_builder.with_vector(&query_vector);

    // Add attribute filters
    for filter in &input.filters {
        let filter_op = match filter.operation.as_str() {
            "equals" => match &filter.value {
                Some(value) => AttributeFilter::equals(&filter.field, value.clone()),
                None => return Err(ApiError::bad_request("Value required for equals operation")),
            },
            "not_equals" => match &filter.value {
                Some(value) => AttributeFilter::not_equals(&filter.field, value.clone()),
                None => {
                    return Err(ApiError::bad_request(
                        "Value required for not_equals operation",
                    ))
                }
            },
            "greater_than" => match &filter.value {
                Some(value) => AttributeFilter::greater_than(&filter.field, value.clone()),
                None => {
                    return Err(ApiError::bad_request(
                        "Value required for greater_than operation",
                    ))
                }
            },
            "less_than" => match &filter.value {
                Some(value) => AttributeFilter::less_than(&filter.field, value.clone()),
                None => {
                    return Err(ApiError::bad_request(
                        "Value required for less_than operation",
                    ))
                }
            },
            "exists" => AttributeFilter::exists(&filter.field),
            _ => {
                return Err(ApiError::bad_request(&format!(
                    "Unsupported filter operation: {}",
                    filter.operation
                )))
            }
        };

        query_builder = query_builder.with_attribute_filter(filter_op);
    }

    // Set limit and similarity threshold
    query_builder = query_builder.with_limit(input.limit);
    query_builder = query_builder.with_threshold(input.threshold);

    // Execute the query
    let query_results = query_builder
        .execute()
        .map_err(|e| ApiError::from_engramdb_error(e))?;

    // Convert to output model
    let mut results = Vec::with_capacity(query_results.len());

    for (node_id, similarity) in query_results {
        // Load the node
        let node = db
            .load(node_id)
            .map_err(|e| ApiError::from_engramdb_error(e))?;

        // Convert to output model
        let node_output = MemoryNodeOutput {
            id: node.id(),
            vector: if input.include_vectors {
                Some(node.vector().to_vec())
            } else {
                None
            },
            attributes: node.attributes().clone(),
            connections: if input.include_connections {
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
            },
            created_at: node.created_at(),
            updated_at: node.updated_at(),
            content: node.content().map(|s| s.to_string()),
        };

        results.push(SearchResultItem {
            node: node_output,
            similarity,
        });
    }

    // Sort results by similarity (highest first)
    results.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap());

    Ok(Json(SearchResults {
        results,
        total: results.len(),
    }))
}
