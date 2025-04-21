//! API module for EngramDB REST server
//! 
//! This module provides a REST API interface to EngramDB using Rocket

#[cfg(feature = "api-server")]
pub mod config;
#[cfg(feature = "api-server")]
pub mod models;
#[cfg(feature = "api-server")]
pub mod routes;
#[cfg(feature = "api-server")]
pub mod error;
#[cfg(feature = "api-server")]
pub mod auth;