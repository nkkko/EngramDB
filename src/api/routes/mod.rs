//! API routes for the EngramDB REST server

mod databases;
mod memory_nodes;
mod connections;
mod search;
mod embeddings;

use rocket::{Build, Rocket};
use rocket_okapi::swagger_ui::{make_swagger_ui, SwaggerUIConfig};

/// Configure all API routes for the Rocket instance
pub fn configure_routes(rocket: Rocket<Build>) -> Rocket<Build> {
    rocket
        .mount(
            "/v1/databases",
            routes![
                databases::list_databases,
                databases::create_database,
                databases::get_database,
                databases::delete_database,
            ],
        )
        .mount(
            "/v1/databases/:database_id/nodes",
            routes![
                memory_nodes::list_nodes,
                memory_nodes::create_node,
                memory_nodes::create_node_from_content,
                memory_nodes::get_node,
                memory_nodes::update_node,
                memory_nodes::delete_node,
            ],
        )
        .mount(
            "/v1/databases/:database_id/search",
            routes![
                search::search_nodes,
            ],
        )
        .mount(
            "/v1/databases/:database_id/nodes/:node_id/connections",
            routes![
                connections::get_node_connections,
                connections::create_connection,
                connections::delete_connection,
            ],
        )
        .mount(
            "/v1",
            routes![
                embeddings::list_models,
                embeddings::generate_embedding,
            ],
        )
        .mount(
            "/swagger",
            make_swagger_ui(&SwaggerUIConfig {
                url: "../openapi.json".to_owned(),
                ..Default::default()
            }),
        )
}

/// Start the API server
pub fn start_server(config: crate::api::config::ApiConfig) -> Result<(), rocket::Error> {
    let figment = rocket::Config::figment()
        .merge(("port", config.port))
        .merge(("address", config.host.clone()));

    let rocket = rocket::custom(figment);
    let rocket = configure_routes(rocket);
    
    // Block until the server shutdowns
    rocket::build()
        .configure(figment)
        .launch()
        .expect("Failed to launch Rocket server");
    
    Ok(())
}