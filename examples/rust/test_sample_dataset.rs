use engramdb::{Database, Result};
use std::path::Path;

mod sample_dataset;

fn print_separator(title: &str) {
    let width = 80;
    let title_len = title.len();
    let padding = (width - title_len - 4) / 2;
    let stars = "*".repeat(padding);
    println!("\n{} {} {}", stars, title, stars);
}

fn main() -> Result<()> {
    // Create a temporary database
    let file_path = Path::new("./temp_sample_dataset.engramdb");
    
    // Test minimal dataset
    print_separator("MINIMAL DATASET");
    let mut db = Database::single_file(file_path)?;
    println!("Loading minimal dataset...");
    let node_ids = sample_dataset::load_minimal_dataset(&mut db)?;
    println!("Successfully loaded {} nodes", node_ids.len());
    
    // Print details of each node
    for id in &node_ids {
        let node = db.load(*id)?;
        let title = if let Some(engramdb::core::AttributeValue::String(title)) = node.get_attribute("title") {
            title.as_str()
        } else {
            "Untitled"
        };
        
        let category = if let Some(engramdb::core::AttributeValue::String(category)) = node.get_attribute("category") {
            category.as_str()
        } else {
            "Uncategorized"
        };
        
        println!("\nNode: {} ({})", title, category);
        println!("  ID: {}", id);
        
        // Print content preview
        if let Some(engramdb::core::AttributeValue::String(content)) = node.get_attribute("content") {
            let content_preview = content.lines().next().unwrap_or("");
            let preview = if content_preview.len() > 60 {
                format!("{}...", &content_preview[..57])
            } else {
                content_preview.to_string()
            };
            println!("  Content preview: {}", preview);
        }
        
        // Print connections
        let connections = db.get_connections(*id, None)?;
        if !connections.is_empty() {
            println!("  Outgoing connections:");
            for conn in connections {
                let target = db.load(conn.target_id)?;
                let target_title = if let Some(engramdb::core::AttributeValue::String(title)) = target.get_attribute("title") {
                    title.as_str()
                } else {
                    "Untitled"
                };
                println!("    → {} ({}, strength: {})", target_title, conn.type_name, conn.strength);
            }
        }
    }
    
    // Cleanup
    let _ = std::fs::remove_file(file_path); // Ignore errors
    
    // Test full dataset
    print_separator("FULL DATASET");
    let mut db = Database::single_file(file_path)?;
    println!("Loading full sample dataset...");
    let node_ids = sample_dataset::load_sample_dataset(&mut db)?;
    println!("Successfully loaded {} nodes", node_ids.len());
    
    // Print details of the first 3 nodes
    for id in node_ids.iter().take(3) {
        let node = db.load(*id)?;
        let title = if let Some(engramdb::core::AttributeValue::String(title)) = node.get_attribute("title") {
            title.as_str()
        } else {
            "Untitled"
        };
        
        let category = if let Some(engramdb::core::AttributeValue::String(category)) = node.get_attribute("category") {
            category.as_str()
        } else {
            "Uncategorized"
        };
        
        println!("\nNode: {} ({})", title, category);
        println!("  ID: {}", id);
        
        // Print content preview
        if let Some(engramdb::core::AttributeValue::String(content)) = node.get_attribute("content") {
            let content_preview = content.lines().next().unwrap_or("");
            let preview = if content_preview.len() > 60 {
                format!("{}...", &content_preview[..57])
            } else {
                content_preview.to_string()
            };
            println!("  Content preview: {}", preview);
        }
        
        // Print connections
        let connections = db.get_connections(*id, None)?;
        if !connections.is_empty() {
            println!("  Outgoing connections:");
            for conn in connections {
                let target = db.load(conn.target_id)?;
                let target_title = if let Some(engramdb::core::AttributeValue::String(title)) = target.get_attribute("title") {
                    title.as_str()
                } else {
                    "Untitled"
                };
                println!("    → {} ({}, strength: {})", target_title, conn.type_name, conn.strength);
            }
        }
    }
    
    // Analyze dataset categories
    println!("\nAnalyzing dataset categories:");
    let mut categories = std::collections::HashMap::new();
    for id in &node_ids {
        let node = db.load(*id)?;
        if let Some(engramdb::core::AttributeValue::String(category)) = node.get_attribute("category") {
            *categories.entry(category.clone()).or_insert(0) += 1;
        }
    }
    
    for (category, count) in categories {
        println!("  {}: {} nodes", category, count);
    }
    
    // Analyze connection types
    println!("\nAnalyzing connection types:");
    let mut connection_types = std::collections::HashMap::new();
    let mut total_connections = 0;
    
    for id in &node_ids {
        let connections = db.get_connections(*id, None)?;
        total_connections += connections.len();
        
        for conn in connections {
            *connection_types.entry(conn.type_name).or_insert(0) += 1;
        }
    }
    
    println!("  Total connections: {}", total_connections);
    for (conn_type, count) in connection_types {
        println!("  {}: {} connections", conn_type, count);
    }
    
    // Cleanup
    let _ = std::fs::remove_file(file_path); // Ignore errors
    println!("\nTest completed successfully!");
    
    Ok(())
}