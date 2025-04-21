//! EngramDB API Server
//!
//! This binary launches a RESTful API server for EngramDB using Rocket

use engramdb::api::config::ApiConfig;
use engramdb::api::routes;
use log::{info, warn};
use std::env;

fn main() {
    // Initialize logging
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    // Load configuration from environment variables
    let config = ApiConfig::from_env();
    info!(
        "Starting EngramDB API server on {}:{}",
        config.host, config.port
    );

    // Display warning about JWT secret
    if env::var("ENGRAMDB_JWT_SECRET").is_err() {
        warn!(
            "No JWT secret provided. Using a generated secret. This is not secure for production."
        );
        warn!("Set the ENGRAMDB_JWT_SECRET environment variable for a stable secret key.");
    }

    // Start the server
    routes::start_server(config).expect("Failed to start server");
}
