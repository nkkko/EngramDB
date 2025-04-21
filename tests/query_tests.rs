use engramdb::{
    core::{AttributeValue, MemoryNode},
    database::Database,
    query::{Filter, QueryBuilder},
    Result,
};
use std::collections::HashSet;
use std::time::{Duration, SystemTime};

/// Create a test database with a diverse set of nodes for query testing
fn create_test_database() -> Result<Database> {
    let db = Database::in_memory();

    // Create nodes with different attributes

    // Group 1: Category A nodes
    for i in 0..5 {
        let mut node = MemoryNode::new(vec![i as f32, 0.0, 0.0]);
        node.set_attribute(
            "category".to_string(),
            AttributeValue::String("A".to_string()),
        );
        node.set_attribute("value".to_string(), AttributeValue::Integer(i as i64));
        node.set_attribute("active".to_string(), AttributeValue::Boolean(i % 2 == 0));
        node.set_attribute("score".to_string(), AttributeValue::Float(i as f32 * 1.5));
        node.set_attribute(
            "tags".to_string(),
            AttributeValue::Array(vec![
                AttributeValue::String("common".to_string()),
                AttributeValue::String(format!("tag{}", i)),
            ]),
        );
        db.save(&node)?;
    }

    // Group 2: Category B nodes
    for i in 5..10 {
        let mut node = MemoryNode::new(vec![0.0, i as f32, 0.0]);
        node.set_attribute(
            "category".to_string(),
            AttributeValue::String("B".to_string()),
        );
        node.set_attribute("value".to_string(), AttributeValue::Integer(i as i64));
        node.set_attribute("active".to_string(), AttributeValue::Boolean(i % 2 == 1));
        node.set_attribute("score".to_string(), AttributeValue::Float(i as f32 * 0.8));
        node.set_attribute(
            "tags".to_string(),
            AttributeValue::Array(vec![
                AttributeValue::String("common".to_string()),
                AttributeValue::String(format!("tag{}", i)),
            ]),
        );
        db.save(&node)?;
    }

    // Group 3: Category C nodes with timestamps
    let now = SystemTime::now();
    for i in 10..15 {
        let mut node = MemoryNode::new(vec![0.0, 0.0, i as f32]);
        node.set_attribute(
            "category".to_string(),
            AttributeValue::String("C".to_string()),
        );
        node.set_attribute("value".to_string(), AttributeValue::Integer(i as i64));
        node.set_attribute("active".to_string(), AttributeValue::Boolean(true));

        // Set creation_time with increasing offsets (newest to oldest)
        let offset = Duration::from_secs(60 * 60 * 24 * (15 - i) as u64); // Days ago
        let timestamp = now
            .checked_sub(offset)
            .unwrap()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap()
            .as_secs();

        node.set_attribute(
            "creation_time".to_string(),
            AttributeValue::Integer(timestamp as i64),
        );
        db.save(&node)?;
    }

    // Create some connections between nodes
    let all_ids = db.list_all()?;
    if all_ids.len() >= 5 {
        // Connect first node to several others
        db.connect(all_ids[0], all_ids[1], "related".to_string(), 0.9)?;
        db.connect(all_ids[0], all_ids[2], "related".to_string(), 0.8)?;
        db.connect(all_ids[0], all_ids[4], "parent".to_string(), 0.7)?;

        // Connect second node to others
        db.connect(all_ids[1], all_ids[3], "child".to_string(), 0.85)?;
    }

    Ok(db)
}

/// Test scenario: Basic attribute filters
///
/// This test verifies that the basic attribute filters
/// (equals, not equals, greater/less than, etc.) work correctly.
#[test]
fn test_query_basic_attribute_filters() -> Result<()> {
    let db = create_test_database()?;

    // Test 1: Equality filter
    let results = db.query().filter("category = 'A'").execute()?;
    assert_eq!(results.len(), 5);
    for result in &results {
        let node = db.load(result.id)?;
        assert_eq!(node.get_attribute("category").unwrap().to_string(), "\"A\"");
    }

    // Test 2: Inequality filter
    let results = db.query().filter("category != 'A'").execute()?;
    assert_eq!(results.len(), 10);
    for result in &results {
        let node = db.load(result.id)?;
        assert_ne!(node.get_attribute("category").unwrap().to_string(), "\"A\"");
    }

    // Test 3: Greater than filter
    let results = db.query().filter("value > 7").execute()?;
    assert!(results.len() >= 7);
    for result in &results {
        let node = db.load(result.id)?;
        let value = node.get_attribute("value").unwrap().to_string();
        let value_int: i64 = value.parse().unwrap();
        assert!(value_int > 7);
    }

    // Test 4: Less than or equal filter
    let results = db.query().filter("value <= 4").execute()?;
    assert!(results.len() >= 5);
    for result in &results {
        let node = db.load(result.id)?;
        let value = node.get_attribute("value").unwrap().to_string();
        let value_int: i64 = value.parse().unwrap();
        assert!(value_int <= 4);
    }

    // Test 5: Boolean filter
    let results = db.query().filter("active = true").execute()?;
    for result in &results {
        let node = db.load(result.id)?;
        assert_eq!(node.get_attribute("active").unwrap().to_string(), "true");
    }

    Ok(())
}

/// Test scenario: Combined filters with AND/OR operators
///
/// This test verifies that multiple filters can be combined
/// with logical operators.
#[test]
fn test_query_combined_filters() -> Result<()> {
    let db = create_test_database()?;

    // Test 1: AND combination
    let results = db
        .query()
        .filter("category = 'A'")
        .filter("value > 2")
        .execute()?;

    assert!(results.len() > 0 && results.len() < 5); // Should be a subset of category A
    for result in &results {
        let node = db.load(result.id)?;

        // Verify both conditions are met
        assert_eq!(node.get_attribute("category").unwrap().to_string(), "\"A\"");

        let value = node.get_attribute("value").unwrap().to_string();
        let value_int: i64 = value.parse().unwrap();
        assert!(value_int > 2);
    }

    // Test 2: OR combination with explicit OR
    let results = db
        .query()
        .filter("(category = 'A') OR (category = 'B')")
        .execute()?;

    assert_eq!(results.len(), 10); // All category A and B nodes
    for result in &results {
        let node = db.load(result.id)?;
        let category = node.get_attribute("category").unwrap().to_string();
        assert!(category == "\"A\"" || category == "\"B\"");
    }

    // Test 3: Complex combination
    let results = db
        .query()
        .filter("(category = 'A' AND value < 2) OR (category = 'B' AND value > 7)")
        .execute()?;

    for result in &results {
        let node = db.load(result.id)?;
        let category = node.get_attribute("category").unwrap().to_string();
        let value = node.get_attribute("value").unwrap().to_string();
        let value_int: i64 = value.parse().unwrap();

        // Verify that either of the combined conditions is met
        assert!((category == "\"A\"" && value_int < 2) || (category == "\"B\"" && value_int > 7));
    }

    Ok(())
}

/// Test scenario: Array and string contains operations
///
/// This test verifies that filters involving arrays and string
/// containment work correctly.
#[test]
fn test_query_contains_operations() -> Result<()> {
    let db = create_test_database()?;

    // Test 1: Array contains
    let results = db.query().filter("tags CONTAINS 'common'").execute()?;

    assert!(results.len() >= 10); // All nodes with tags should match

    // Test 2: Array contains specific tag
    let results = db.query().filter("tags CONTAINS 'tag3'").execute()?;

    assert_eq!(results.len(), 1); // Only one node has tag3

    // Test 3: String contains
    let results = db.query().filter("category CONTAINS 'A'").execute()?;

    assert_eq!(results.len(), 5); // All category A nodes

    // Test 4: Combined contains operations
    let results = db
        .query()
        .filter("tags CONTAINS 'common' AND category CONTAINS 'B'")
        .execute()?;

    assert_eq!(results.len(), 5); // All category B nodes with common tag

    Ok(())
}

/// Test scenario: Sorting and pagination
///
/// This test verifies that query results can be sorted and paginated.
#[test]
fn test_query_sorting_and_pagination() -> Result<()> {
    let db = create_test_database()?;

    // Test 1: Sort by value ascending
    let results = db.query().sort_by("value", true).execute()?;

    // Verify ascending order
    for i in 1..results.len() {
        let node1 = db.load(results[i - 1].id)?;
        let node2 = db.load(results[i].id)?;

        let value1 = node1.get_attribute("value").unwrap().to_string();
        let value2 = node2.get_attribute("value").unwrap().to_string();

        let val1: i64 = value1.parse().unwrap();
        let val2: i64 = value2.parse().unwrap();

        assert!(val1 <= val2);
    }

    // Test 2: Sort by value descending
    let results = db.query().sort_by("value", false).execute()?;

    // Verify descending order
    for i in 1..results.len() {
        let node1 = db.load(results[i - 1].id)?;
        let node2 = db.load(results[i].id)?;

        let value1 = node1.get_attribute("value").unwrap().to_string();
        let value2 = node2.get_attribute("value").unwrap().to_string();

        let val1: i64 = value1.parse().unwrap();
        let val2: i64 = value2.parse().unwrap();

        assert!(val1 >= val2);
    }

    // Test 3: Pagination - first page
    let page_size = 5;
    let first_page = db
        .query()
        .sort_by("value", true)
        .limit(page_size)
        .execute()?;

    assert_eq!(first_page.len(), page_size);

    // Get lowest value for verification
    let lowest_values: HashSet<_> = first_page
        .iter()
        .map(|r| {
            let node = db.load(r.id).unwrap();
            let value = node.get_attribute("value").unwrap().to_string();
            value.parse::<i64>().unwrap()
        })
        .collect();

    // Test 4: Pagination - second page
    let second_page = db
        .query()
        .sort_by("value", true)
        .limit(page_size)
        .offset(page_size)
        .execute()?;

    assert!(second_page.len() <= page_size);

    // Verify second page has higher values than first page
    for result in &second_page {
        let node = db.load(result.id)?;
        let value = node.get_attribute("value").unwrap().to_string();
        let value_int: i64 = value.parse().unwrap();

        // Value should not be in the first page
        assert!(!lowest_values.contains(&value_int));
    }

    Ok(())
}

/// Test scenario: Temporal queries
///
/// This test verifies that queries involving timestamps and
/// temporal relationships work correctly.
#[test]
fn test_query_temporal_filters() -> Result<()> {
    let db = create_test_database()?;

    // Get current timestamp
    let now = SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap()
        .as_secs() as i64;

    // Test 1: Nodes created within the last 3 days
    let three_days_ago = now - (60 * 60 * 24 * 3);

    let results = db
        .query()
        .filter(&format!("creation_time > {}", three_days_ago))
        .execute()?;

    // We should have some nodes that were "created" recently
    assert!(results.len() > 0);

    // Verify their timestamps are indeed within the time window
    for result in &results {
        let node = db.load(result.id)?;
        if let Some(time_attr) = node.get_attribute("creation_time") {
            let time = time_attr.to_string().parse::<i64>().unwrap();
            assert!(time > three_days_ago);
        }
    }

    // Test 2: Nodes created between 5 and 2 days ago
    let five_days_ago = now - (60 * 60 * 24 * 5);
    let two_days_ago = now - (60 * 60 * 24 * 2);

    let results = db
        .query()
        .filter(&format!(
            "creation_time > {} AND creation_time < {}",
            five_days_ago, two_days_ago
        ))
        .execute()?;

    // Verify the range constraint
    for result in &results {
        let node = db.load(result.id)?;
        if let Some(time_attr) = node.get_attribute("creation_time") {
            let time = time_attr.to_string().parse::<i64>().unwrap();
            assert!(time > five_days_ago && time < two_days_ago);
        }
    }

    Ok(())
}

/// Test scenario: Connection-based queries
///
/// This test verifies that queries can filter by node connections
/// and traverse the graph structure.
#[test]
fn test_query_connection_filters() -> Result<()> {
    let db = create_test_database()?;

    // Get first node ID which should have connections
    let root_id = db.list_all()?[0];

    // Test 1: Get nodes connected to the root
    let connected_nodes = db.get_connections(root_id, None)?;
    assert!(!connected_nodes.is_empty()); // Should have some connections

    // Extract target IDs from connections
    let target_ids: HashSet<_> = connected_nodes.iter().map(|conn| conn.target_id).collect();

    // Test 2: Filter by specific relationship type
    let related_conns = db.get_connections(root_id, Some("related".to_string()))?;

    // Verify relationship type filter
    for conn in &related_conns {
        assert_eq!(conn.relationship_type, "related");
        assert!(target_ids.contains(&conn.target_id));
    }

    // Test 3: Two-hop traversal
    // First find node1's connections
    if let Some(&first_connected_id) = target_ids.iter().next() {
        // Then find connections from this node
        let second_hop = db.get_connections(first_connected_id, None)?;

        // Some of the first_connected nodes should have their own connections
        if !second_hop.is_empty() {
            let second_hop_ids: HashSet<_> = second_hop.iter().map(|conn| conn.target_id).collect();

            // Verify these are different from the direct connections
            for id in &second_hop_ids {
                if *id != root_id {
                    // Skip back-references
                    // This is a two-hop connection
                    assert!(!target_ids.contains(id) || *id == root_id);
                }
            }
        }
    }

    Ok(())
}

/// Test scenario: Query with vector similarity and filters
///
/// This test verifies that vector similarity searches can be
/// combined with attribute filters.
#[test]
fn test_query_vector_similarity_with_filters() -> Result<()> {
    let db = create_test_database()?;

    // Create a query vector
    let query_vector = vec![0.5, 0.3, 0.2];

    // Test 1: Basic vector search
    let vector_results = db.search_similar(&query_vector, 5, 0.0, None, None)?;
    assert!(!vector_results.is_empty());

    // Test 2: Vector search with category filter
    let filtered_results = db.search_similar(
        &query_vector,
        5,
        0.0,
        Some("category = 'A'".to_string()),
        None,
    )?;

    // Verify all results have category A
    for result in &filtered_results {
        let node = db.load(result.id)?;
        assert_eq!(node.get_attribute("category").unwrap().to_string(), "\"A\"");
    }

    // Test 3: Vector search with complex filter
    let complex_results = db.search_similar(
        &query_vector,
        10,
        0.0,
        Some("(category = 'A' AND value > 2) OR (category = 'B' AND active = true)".to_string()),
        None,
    )?;

    // Verify all results match the complex filter
    for result in &complex_results {
        let node = db.load(result.id)?;
        let category = node.get_attribute("category").unwrap().to_string();

        if category == "\"A\"" {
            let value = node.get_attribute("value").unwrap().to_string();
            let value_int: i64 = value.parse().unwrap();
            assert!(value_int > 2);
        } else if category == "\"B\"" {
            assert_eq!(node.get_attribute("active").unwrap().to_string(), "true");
        } else {
            panic!("Unexpected category: {}", category);
        }
    }

    Ok(())
}

/// Test scenario: Edge cases
///
/// This test verifies that the query system handles edge cases
/// correctly (empty results, invalid filters, etc.)
#[test]
fn test_query_edge_cases() -> Result<()> {
    let db = create_test_database()?;

    // Test 1: Filter that matches nothing
    let empty_results = db
        .query()
        .filter("category = 'Z'") // No nodes have category Z
        .execute()?;

    assert!(empty_results.is_empty());

    // Test 2: Filter on non-existent attribute
    let no_attr_results = db
        .query()
        .filter("non_existent_field = 'something'")
        .execute()?;

    assert!(no_attr_results.is_empty());

    // Test 3: Very high offset (beyond available results)
    let high_offset_results = db
        .query()
        .offset(1000) // Way more than available nodes
        .execute()?;

    assert!(high_offset_results.is_empty());

    // Test 4: Zero limit
    let zero_limit_results = db.query().limit(0).execute()?;

    assert!(zero_limit_results.is_empty());

    // Test 5: Huge limit
    let huge_limit_results = db
        .query()
        .limit(1000) // Way more than available nodes
        .execute()?;

    // Should return all available nodes, but not crash
    assert_eq!(huge_limit_results.len(), db.len()?);

    Ok(())
}
