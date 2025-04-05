//! Core memory representation and data structures
//!
//! This module contains the fundamental data structures for representing
//! memories in the EngramDB system.

mod access_history;
pub mod connection;
mod memory_node;
mod temporal_layer;

pub use access_history::AccessHistory;
pub use connection::Connection;
pub use memory_node::MemoryNode;
pub use temporal_layer::TemporalLayer;

use serde::{Deserialize, Serialize};

/// Represents different types of attribute values that can be stored in a MemoryNode
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum AttributeValue {
    /// String value
    String(String),
    /// Integer value
    Integer(i64),
    /// Float value
    Float(f64),
    /// Boolean value
    Boolean(bool),
    /// List of attribute values
    List(Vec<AttributeValue>),
    /// Nested map of attribute values
    Map(std::collections::HashMap<String, AttributeValue>),
}
