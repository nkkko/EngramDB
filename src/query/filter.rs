use crate::core::{AttributeValue, MemoryNode};

/// Filter for memory node attributes
#[derive(Debug, Clone)]
pub struct AttributeFilter {
    /// The attribute key to filter on
    key: String,

    /// The comparison operation to apply
    operation: AttributeOperation,
}

/// Types of attribute comparison operations
#[derive(Debug, Clone)]
enum AttributeOperation {
    /// Equal to the specified value
    Equals(AttributeValue),

    /// Not equal to the specified value
    NotEquals(AttributeValue),

    /// Greater than the specified value (for numeric types)
    GreaterThan(AttributeValue),

    /// Less than the specified value (for numeric types)
    LessThan(AttributeValue),

    /// Contains the specified string (for string types)
    Contains(String),

    /// The attribute exists
    Exists,

    /// The attribute does not exist
    NotExists,
}

impl AttributeFilter {
    /// Creates a filter that checks if an attribute equals a value
    pub fn equals(key: String, value: AttributeValue) -> Self {
        Self {
            key,
            operation: AttributeOperation::Equals(value),
        }
    }

    /// Creates a filter that checks if an attribute does not equal a value
    pub fn not_equals(key: String, value: AttributeValue) -> Self {
        Self {
            key,
            operation: AttributeOperation::NotEquals(value),
        }
    }

    /// Creates a filter that checks if an attribute is greater than a value
    pub fn greater_than(key: String, value: AttributeValue) -> Self {
        Self {
            key,
            operation: AttributeOperation::GreaterThan(value),
        }
    }

    /// Creates a filter that checks if an attribute is less than a value
    pub fn less_than(key: String, value: AttributeValue) -> Self {
        Self {
            key,
            operation: AttributeOperation::LessThan(value),
        }
    }

    /// Creates a filter that checks if a string attribute contains a substring
    pub fn contains(key: String, substring: String) -> Self {
        Self {
            key,
            operation: AttributeOperation::Contains(substring),
        }
    }

    /// Creates a filter that checks if an attribute exists
    pub fn exists(key: String) -> Self {
        Self {
            key,
            operation: AttributeOperation::Exists,
        }
    }

    /// Creates a filter that checks if an attribute does not exist
    pub fn not_exists(key: String) -> Self {
        Self {
            key,
            operation: AttributeOperation::NotExists,
        }
    }

    /// Applies the filter to a memory node
    ///
    /// # Arguments
    ///
    /// * `node` - The memory node to filter
    ///
    /// # Returns
    ///
    /// true if the node passes the filter, false otherwise
    pub fn apply(&self, node: &MemoryNode) -> bool {
        let attribute = node.get_attribute(&self.key);

        match &self.operation {
            AttributeOperation::Equals(value) => {
                if let Some(attr) = attribute {
                    attr == value
                } else {
                    false
                }
            }
            AttributeOperation::NotEquals(value) => {
                if let Some(attr) = attribute {
                    attr != value
                } else {
                    true
                }
            }
            AttributeOperation::GreaterThan(value) => {
                if let Some(attr) = attribute {
                    match (attr, value) {
                        (AttributeValue::Integer(a), AttributeValue::Integer(b)) => a > b,
                        (AttributeValue::Float(a), AttributeValue::Float(b)) => a > b,
                        (AttributeValue::Integer(a), AttributeValue::Float(b)) => (*a as f64) > *b,
                        (AttributeValue::Float(a), AttributeValue::Integer(b)) => *a > (*b as f64),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            AttributeOperation::LessThan(value) => {
                if let Some(attr) = attribute {
                    match (attr, value) {
                        (AttributeValue::Integer(a), AttributeValue::Integer(b)) => a < b,
                        (AttributeValue::Float(a), AttributeValue::Float(b)) => a < b,
                        (AttributeValue::Integer(a), AttributeValue::Float(b)) => (*a as f64) < *b,
                        (AttributeValue::Float(a), AttributeValue::Integer(b)) => *a < (*b as f64),
                        _ => false,
                    }
                } else {
                    false
                }
            }
            AttributeOperation::Contains(substring) => {
                if let Some(AttributeValue::String(s)) = attribute {
                    s.contains(substring)
                } else {
                    false
                }
            }
            AttributeOperation::Exists => attribute.is_some(),
            AttributeOperation::NotExists => attribute.is_none(),
        }
    }
}

/// Filter for memory node temporal aspects
#[derive(Debug, Clone)]
pub struct TemporalFilter {
    /// The temporal operation to apply
    operation: TemporalOperation,
}

/// Types of temporal filtering operations
#[derive(Debug, Clone)]
enum TemporalOperation {
    /// Created before a specific timestamp
    Before(u64),

    /// Created after a specific timestamp
    After(u64),

    /// Created between two timestamps
    Between(u64, u64),

    /// Created after a relative time (seconds ago from now)
    WithinLast(u64),
}

impl TemporalFilter {
    /// Creates a filter for memories created before a timestamp
    pub fn before(timestamp: u64) -> Self {
        Self {
            operation: TemporalOperation::Before(timestamp),
        }
    }

    /// Creates a filter for memories created after a timestamp
    pub fn after(timestamp: u64) -> Self {
        Self {
            operation: TemporalOperation::After(timestamp),
        }
    }

    /// Creates a filter for memories created between two timestamps
    pub fn between(start: u64, end: u64) -> Self {
        Self {
            operation: TemporalOperation::Between(start, end),
        }
    }

    /// Creates a filter for memories created within the last N seconds
    pub fn within_last(seconds: u64) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let start = now.saturating_sub(seconds);

        Self {
            operation: TemporalOperation::WithinLast(start),
        }
    }

    /// Applies the filter to a memory node
    ///
    /// # Arguments
    ///
    /// * `node` - The memory node to filter
    ///
    /// # Returns
    ///
    /// true if the node passes the filter, false otherwise
    pub fn apply(&self, node: &MemoryNode) -> bool {
        let timestamp = node.creation_timestamp();

        match self.operation {
            TemporalOperation::Before(ts) => timestamp < ts,
            TemporalOperation::After(ts) => timestamp > ts,
            TemporalOperation::Between(start, end) => timestamp >= start && timestamp <= end,
            TemporalOperation::WithinLast(start) => timestamp >= start,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use uuid::Uuid;

    #[test]
    fn test_attribute_filter_equals() {
        let mut node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        node.set_attribute(
            "name".to_string(),
            AttributeValue::String("test".to_string()),
        );

        // Test equals (match)
        let filter = AttributeFilter::equals(
            "name".to_string(),
            AttributeValue::String("test".to_string()),
        );
        assert!(filter.apply(&node));

        // Test equals (no match)
        let filter = AttributeFilter::equals(
            "name".to_string(),
            AttributeValue::String("other".to_string()),
        );
        assert!(!filter.apply(&node));

        // Test equals (attribute doesn't exist)
        let filter = AttributeFilter::equals(
            "nonexistent".to_string(),
            AttributeValue::String("test".to_string()),
        );
        assert!(!filter.apply(&node));
    }

    #[test]
    fn test_attribute_filter_numeric_comparison() {
        let mut node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        node.set_attribute("importance".to_string(), AttributeValue::Float(0.8));
        node.set_attribute("count".to_string(), AttributeValue::Integer(42));

        // Test greater_than (match)
        let filter =
            AttributeFilter::greater_than("importance".to_string(), AttributeValue::Float(0.5));
        assert!(filter.apply(&node));

        // Test greater_than (no match)
        let filter =
            AttributeFilter::greater_than("importance".to_string(), AttributeValue::Float(0.9));
        assert!(!filter.apply(&node));

        // Test less_than (match)
        let filter = AttributeFilter::less_than("count".to_string(), AttributeValue::Integer(50));
        assert!(filter.apply(&node));

        // Test mixed numeric types (float attribute, integer comparison)
        let filter =
            AttributeFilter::greater_than("importance".to_string(), AttributeValue::Integer(0));
        assert!(filter.apply(&node));
    }

    #[test]
    fn test_attribute_filter_contains() {
        let mut node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        node.set_attribute(
            "description".to_string(),
            AttributeValue::String("This is a test memory".to_string()),
        );

        // Test contains (match)
        let filter = AttributeFilter::contains("description".to_string(), "test".to_string());
        assert!(filter.apply(&node));

        // Test contains (no match)
        let filter =
            AttributeFilter::contains("description".to_string(), "nonexistent".to_string());
        assert!(!filter.apply(&node));

        // Test contains on non-string attribute
        node.set_attribute("count".to_string(), AttributeValue::Integer(42));
        let filter = AttributeFilter::contains("count".to_string(), "4".to_string());
        assert!(!filter.apply(&node));
    }

    #[test]
    fn test_attribute_filter_exists() {
        let mut node = MemoryNode::new(vec![0.1, 0.2, 0.3]);
        node.set_attribute(
            "name".to_string(),
            AttributeValue::String("test".to_string()),
        );

        // Test exists (match)
        let filter = AttributeFilter::exists("name".to_string());
        assert!(filter.apply(&node));

        // Test exists (no match)
        let filter = AttributeFilter::exists("nonexistent".to_string());
        assert!(!filter.apply(&node));

        // Test not_exists (match)
        let filter = AttributeFilter::not_exists("nonexistent".to_string());
        assert!(filter.apply(&node));

        // Test not_exists (no match)
        let filter = AttributeFilter::not_exists("name".to_string());
        assert!(!filter.apply(&node));
    }

    #[test]
    fn test_temporal_filter() {
        // Create test nodes with specific timestamps using the test helper API

        // We'll use fixed UUIDs for deterministic tests
        let id1 = Uuid::parse_str("00000000-0000-0000-0000-000000000001").unwrap();
        let id2 = Uuid::parse_str("00000000-0000-0000-0000-000000000002").unwrap();
        let id3 = Uuid::parse_str("00000000-0000-0000-0000-000000000003").unwrap();

        // Create test nodes with the helper method from MemoryNode
        let early_node = MemoryNode::test_with_id_and_timestamp(id1, vec![0.1, 0.2, 0.3], 1000);
        let middle_node = MemoryNode::test_with_id_and_timestamp(id2, vec![0.1, 0.2, 0.3], 1500);
        let late_node = MemoryNode::test_with_id_and_timestamp(id3, vec![0.1, 0.2, 0.3], 2000);

        // Test various temporal filters

        // Test before
        let before_filter = TemporalFilter::before(1500);
        assert!(before_filter.apply(&early_node));
        assert!(!before_filter.apply(&middle_node));
        assert!(!before_filter.apply(&late_node));

        // Test after
        let after_filter = TemporalFilter::after(1500);
        assert!(!after_filter.apply(&early_node));
        assert!(!after_filter.apply(&middle_node));
        assert!(after_filter.apply(&late_node));

        // Test between
        let between_filter = TemporalFilter::between(1200, 1800);
        assert!(!between_filter.apply(&early_node));
        assert!(between_filter.apply(&middle_node));
        assert!(!between_filter.apply(&late_node));
    }
}
