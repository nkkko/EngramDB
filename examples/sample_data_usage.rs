use engramdb::{Database, Result};
use std::path::Path;

mod sample_dataset;

fn main() -> Result<()> {
    // Create a database with single-file storage for this example
    println!("Creating database with sample dataset...");
    let file_path = Path::new("./sample_data_example.engramdb");
    let mut db = Database::single_file(file_path)?;
    
    // Load the minimal sample dataset
    println!("Loading sample dataset...");
    let node_ids = sample_dataset::load_minimal_dataset(&mut db)?;
    println!("Loaded {} nodes into the database", node_ids.len());
    
    // Display all nodes in the database
    println!("\nListing all memory nodes:");
    for id in db.list_all()? {
        let node = db.load(id)?;
        let title = node.get_attribute("title").and_then(|attr| {
            if let engramdb::core::AttributeValue::String(s) = attr {
                Some(s.as_str())
            } else {
                None
            }
        }).unwrap_or("Untitled");
        
        let category = node.get_attribute("category").and_then(|attr| {
            if let engramdb::core::AttributeValue::String(s) = attr {
                Some(s.as_str())
            } else {
                None
            }
        }).unwrap_or("Uncategorized");
        
        println!("  Node {}: {} ({})", id, title, category);
        
        // List its connections
        let connections = db.get_connections(id, None)?;
        if !connections.is_empty() {
            println!("    Connections:");
            for conn in connections {
                println!("      -> {} ({}, strength: {})", 
                    conn.target_id, conn.type_name, conn.strength);
            }
        }
    }
    
    // Perform a vector search
    println!("\nPerforming vector search for concept similar to 'requirements'...");
    let search_results = db.search_similar(&[0.9, 0.2, 0.1], 3, 0.0, None, None)?;
    println!("Search results:");
    for (id, similarity) in search_results {
        let node = db.load(id)?;
        let title = node.get_attribute("title").and_then(|attr| {
            if let engramdb::core::AttributeValue::String(s) = attr {
                Some(s.as_str())
            } else {
                None
            }
        }).unwrap_or("Untitled");
        
        println!("  {}: similarity {:.2}", title, similarity);
    }
    
    // Load the larger sample dataset (optional)
    let load_full_dataset = std::env::var("LOAD_FULL_DATASET").map(|v| v == "1" || v.to_lowercase() == "true").unwrap_or(false);
    if load_full_dataset {
        println!("\nLoading full AI bugfix workflow dataset...");
        db.clear_all()?;
        let node_ids = sample_dataset::load_sample_dataset(&mut db)?;
        println!("Loaded {} nodes from the full dataset", node_ids.len());
        
        // Print a few representative nodes
        println!("\nSample nodes from full dataset:");
        let nodes_to_show = 3;
        for id in node_ids.iter().take(nodes_to_show) {
            let node = db.load(*id)?;
            let title = node.get_attribute("title").and_then(|attr| {
                if let engramdb::core::AttributeValue::String(s) = attr {
                    Some(s.as_str())
                } else {
                    None
                }
            }).unwrap_or("Untitled");
            
            let category = node.get_attribute("category").and_then(|attr| {
                if let engramdb::core::AttributeValue::String(s) = attr {
                    Some(s.as_str())
                } else {
                    None
                }
            }).unwrap_or("Uncategorized");
            
            println!("  Node {}: {} ({})", id, title, category);
        }
        
        println!("... and {} more nodes", node_ids.len() - nodes_to_show);
    }
    
    println!("\nExample completed successfully!");
    Ok(())
}