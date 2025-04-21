use clap::{Parser, Subcommand};
use chrono::Local;
use engramdb::core::{AttributeValue, MemoryNode};
use engramdb::database::{Database, DatabaseConfig, StorageType};
use engramdb::vector::VectorIndexConfig;
use engramdb::error::EngramDbError;
use engramdb::Result;
use std::io::{self, BufRead, Write};
use std::path::PathBuf;
use std::str::FromStr;
use uuid::Uuid;

/// Command line arguments parsing
#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Path to database file or directory
    #[arg(short, long, value_name = "FILE")]
    database: Option<PathBuf>,

    /// Use in-memory database (will be lost when the program exits)
    #[arg(short, long)]
    memory: bool,

    /// Storage type: multi-file, single-file, or memory
    #[arg(short, long, default_value = "multi-file")]
    storage_type: String,
    
    /// Command to execute
    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// List all memories
    List,
    
    /// View a specific memory
    View {
        /// Memory ID
        id: String,
    },
    
    /// Create a new memory
    Create {
        /// Description of the memory
        #[arg(short, long)]
        description: String,
        
        /// Importance value (0.0-1.0)
        #[arg(short, long, default_value = "0.5")]
        importance: f32,
        
        /// Tags (comma separated)
        #[arg(short, long, default_value = "")]
        tags: String,
    },
    
    /// Edit an existing memory
    Edit {
        /// Memory ID
        id: String,
        
        /// Description of the memory
        #[arg(short, long)]
        description: Option<String>,
        
        /// Importance value (0.0-1.0)
        #[arg(short, long)]
        importance: Option<f32>,
        
        /// Tags (comma separated)
        #[arg(short, long)]
        tags: Option<String>,
    },
    
    /// Delete a memory
    Delete {
        /// Memory ID
        id: String,
        
        /// Skip confirmation
        #[arg(short, long)]
        force: bool,
    },
    
    /// Search for memories
    Search {
        /// Search query
        query: String,
    },
    
    /// Add a connection between memories
    Connect {
        /// Source memory ID
        #[arg(short, long)]
        source: String,
        
        /// Target memory ID
        #[arg(short, long)]
        target: String,
        
        /// Relationship type (Association, Causation, PartOf, Contains, Sequence, or custom)
        #[arg(short, long, default_value = "Association")]
        relationship: String,
        
        /// Connection strength (0.0-1.0)
        #[arg(short, long, default_value = "0.8")]
        strength: f32,
    },
    
    /// Remove a connection between memories
    Disconnect {
        /// Source memory ID
        #[arg(short, long)]
        source: String,
        
        /// Target memory ID
        #[arg(short, long)]
        target: String,
    },
    
    /// Interactive shell mode
    Shell,
}

fn print_memory(id: Uuid, memory: &MemoryNode) {
    println!("╔══════════════════════════════════════════════════════════════");
    println!("║ Memory ID: {}", id);
    println!("╠══════════════════════════════════════════════════════════════");
    
    // Print attributes
    println!("║ Attributes:");
    for (key, value) in memory.attributes() {
        if key.starts_with('_') {
            continue; // Skip internal attributes
        }
        
        let value_str = match value {
            AttributeValue::String(s) => s.clone(),
            AttributeValue::Integer(i) => i.to_string(),
            AttributeValue::Float(f) => f.to_string(),
            AttributeValue::Boolean(b) => b.to_string(),
            AttributeValue::List(v) => format!("[List: {} elements]", v.len()),
            AttributeValue::Map(_) => "[Map]".to_string(),
        };
        
        println!("║   {}: {}", key, value_str);
    }
    
    // Print embeddings preview
    let preview = if let Some(embeddings) = memory.embeddings() {
        if embeddings.len() <= 5 {
            format!("{:?}", embeddings)
        } else {
            format!("[{:.2}, {:.2}, {:.2}, ... {} more values]", 
                    embeddings[0], embeddings[1], embeddings[2], embeddings.len() - 3)
        }
    } else if memory.is_multi_vector() {
        "Multi-vector embeddings (ColBERT/ColPali-style)".to_string()
    } else {
        "No embeddings".to_string()
    };
    
    println!("║ Embeddings: {}", preview);
    
    // Print connections
    println!("║ Connections:");
    let connections = memory.connections();
    if connections.is_empty() {
        println!("║   None");
    } else {
        for conn in connections {
            let rel_type = match conn.relationship_type() {
                engramdb::RelationshipType::Association => "Association",
                engramdb::RelationshipType::Causation => "Causation",
                engramdb::RelationshipType::PartOf => "PartOf",
                engramdb::RelationshipType::Contains => "Contains",
                engramdb::RelationshipType::Sequence => "Sequence",
                engramdb::RelationshipType::Custom(s) => s,
            };
            
            println!("║   → {} ({}, strength: {:.2})", 
                    conn.target_id(), rel_type, conn.strength());
        }
    }
    
    println!("╚══════════════════════════════════════════════════════════════");
}

fn list_memories(db: &Database) -> Result<()> {
    let memory_ids = db.list_all()?;
    
    if memory_ids.is_empty() {
        println!("No memories found in database.");
        return Ok(());
    }
    
    println!("Found {} memories:", memory_ids.len());
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════");
    println!("║ {:^36} │ {:^36} │ {}", "ID", "Description", "Created");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════");
    
    // Load and sort memories by creation time
    let mut memories = Vec::new();
    for id in memory_ids {
        let memory = db.load(id)?;
        memories.push((id, memory));
    }
    
    // Sort by creation time (newest first)
    memories.sort_by(|a, b| {
        let a_time = a.1.creation_timestamp();
        let b_time = b.1.creation_timestamp();
        b_time.cmp(&a_time)
    });
    
    for (id, memory) in memories {
        let description = memory.get_attribute("description")
            .and_then(|v| match v {
                AttributeValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("No description");
        
        // Format creation time
        let datetime = chrono::DateTime::from_timestamp(memory.creation_timestamp() as i64, 0)
            .unwrap_or_else(|| chrono::DateTime::UNIX_EPOCH);
        let formatted_time = datetime.format("%Y-%m-%d %H:%M:%S").to_string();
        
        println!("║ {} │ {:<36} │ {}", id, 
                 if description.len() > 36 {
                     format!("{}...", &description[0..33])
                 } else {
                     description.to_string()
                 },
                 formatted_time);
    }
    
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════");
    Ok(())
}

fn view_memory(db: &Database, id_str: &str) -> Result<()> {
    let id = match Uuid::from_str(id_str) {
        Ok(id) => id,
        Err(_) => return Err(EngramDbError::Validation(format!("Invalid UUID: {}", id_str))),
    };
    
    let memory = db.load(id)?;
    print_memory(id, &memory);
    
    Ok(())
}

fn create_memory(db: &mut Database, description: &str, importance: f32, tags: &str) -> Result<Uuid> {
    // Create a simple vector embedding (for demo purposes)
    // In a real app, you'd generate this from text using an embedding model
    let embeddings = vec![
        0.1, 0.2, 0.3, 0.4, 0.5, 
        0.1, 0.2, 0.3, 0.4, 0.5,
    ];
    
    let mut memory = MemoryNode::new(embeddings);
    
    // Add attributes
    memory.set_attribute(
        "description".to_string(),
        AttributeValue::String(description.to_string()),
    );
    
    memory.set_attribute(
        "importance".to_string(),
        AttributeValue::Float(importance.into()),
    );
    
    if !tags.is_empty() {
        memory.set_attribute(
            "tags".to_string(),
            AttributeValue::String(tags.to_string()),
        );
    }
    
    // Add timestamp for sorting
    let timestamp = Local::now().timestamp() as i64;
    memory.set_attribute(
        "creation_timestamp".to_string(), 
        AttributeValue::Integer(timestamp)
    );
    
    let id = memory.id();
    db.save(&memory)?;
    
    println!("Memory created with ID: {}", id);
    Ok(id)
}

fn edit_memory(
    db: &mut Database, 
    id_str: &str, 
    description: Option<&str>, 
    importance: Option<f32>, 
    tags: Option<&str>
) -> Result<()> {
    let id = match Uuid::from_str(id_str) {
        Ok(id) => id,
        Err(_) => return Err(EngramDbError::Validation(format!("Invalid UUID: {}", id_str))),
    };
    
    let mut memory = db.load(id)?;
    
    // Update attributes if provided
    if let Some(desc) = description {
        memory.set_attribute(
            "description".to_string(),
            AttributeValue::String(desc.to_string()),
        );
    }
    
    if let Some(imp) = importance {
        memory.set_attribute(
            "importance".to_string(),
            AttributeValue::Float(imp.into()),
        );
    }
    
    if let Some(t) = tags {
        memory.set_attribute(
            "tags".to_string(),
            AttributeValue::String(t.to_string()),
        );
    }
    
    db.save(&memory)?;
    
    println!("Memory updated: {}", id);
    print_memory(id, &memory);
    
    Ok(())
}

fn delete_memory(db: &mut Database, id_str: &str, force: bool) -> Result<()> {
    let id = match Uuid::from_str(id_str) {
        Ok(id) => id,
        Err(_) => return Err(EngramDbError::Validation(format!("Invalid UUID: {}", id_str))),
    };
    
    if !force {
        // Show memory and ask for confirmation
        let memory = db.load(id)?;
        print_memory(id, &memory);
        
        print!("Are you sure you want to delete this memory? (y/N): ");
        io::stdout().flush().unwrap();
        
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        
        if !input.trim().eq_ignore_ascii_case("y") {
            println!("Deletion cancelled.");
            return Ok(());
        }
    }
    
    db.delete(id)?;
    println!("Memory deleted: {}", id);
    
    Ok(())
}

fn search_memories(db: &Database, query: &str) -> Result<()> {
    let memory_ids = db.list_all()?;
    
    if memory_ids.is_empty() {
        println!("No memories found in database.");
        return Ok(());
    }
    
    let mut results = Vec::new();
    let query_lower = query.to_lowercase();
    
    for id in memory_ids {
        let memory = db.load(id)?;
        let mut matched = false;
        
        // Search in all string attributes
        for (_, value) in memory.attributes() {
            if let AttributeValue::String(s) = value {
                if s.to_lowercase().contains(&query_lower) {
                    matched = true;
                    break;
                }
            }
        }
        
        if matched {
            results.push((id, memory));
        }
    }
    
    if results.is_empty() {
        println!("No results found for query: '{}'", query);
        return Ok(());
    }
    
    println!("Found {} results for query '{}':", results.len(), query);
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════");
    println!("║ {:^36} │ {:^36} │ {}", "ID", "Description", "Created");
    println!("╠══════════════════════════════════════════════════════════════════════════════════════════");
    
    for (id, memory) in results {
        let description = memory.get_attribute("description")
            .and_then(|v| match v {
                AttributeValue::String(s) => Some(s.as_str()),
                _ => None,
            })
            .unwrap_or("No description");
        
        // Format creation time
        let datetime = chrono::DateTime::from_timestamp(memory.creation_timestamp() as i64, 0)
            .unwrap_or_else(|| chrono::DateTime::UNIX_EPOCH);
        let formatted_time = datetime.format("%Y-%m-%d %H:%M:%S").to_string();
        
        println!("║ {} │ {:<36} │ {}", id, 
                 if description.len() > 36 {
                     format!("{}...", &description[0..33])
                 } else {
                     description.to_string()
                 },
                 formatted_time);
    }
    
    println!("╚══════════════════════════════════════════════════════════════════════════════════════════");
    Ok(())
}

fn add_connection(
    db: &mut Database, 
    source_str: &str, 
    target_str: &str, 
    relationship: &str, 
    strength: f32
) -> Result<()> {
    let source_id = match Uuid::from_str(source_str) {
        Ok(id) => id,
        Err(_) => return Err(EngramDbError::Validation(format!("Invalid source UUID: {}", source_str))),
    };
    
    let target_id = match Uuid::from_str(target_str) {
        Ok(id) => id,
        Err(_) => return Err(EngramDbError::Validation(format!("Invalid target UUID: {}", target_str))),
    };
    
    db.connect(source_id, target_id, relationship.to_string(), strength)?;
    
    println!("Connection added from {} to {} (type: {}, strength: {:.2})", 
             source_id, target_id, relationship, strength);
    
    Ok(())
}

fn remove_connection(db: &mut Database, source_str: &str, target_str: &str) -> Result<()> {
    let source_id = match Uuid::from_str(source_str) {
        Ok(id) => id,
        Err(_) => return Err(EngramDbError::Validation(format!("Invalid source UUID: {}", source_str))),
    };
    
    let target_id = match Uuid::from_str(target_str) {
        Ok(id) => id,
        Err(_) => return Err(EngramDbError::Validation(format!("Invalid target UUID: {}", target_str))),
    };
    
    let removed = db.disconnect(source_id, target_id)?;
    
    if removed {
        println!("Connection removed from {} to {}", source_id, target_id);
    } else {
        println!("No connection found from {} to {}", source_id, target_id);
    }
    
    Ok(())
}

fn interactive_shell(db: &mut Database) -> Result<()> {
    println!("EngramDB Interactive Shell");
    println!("Type 'help' for a list of commands, 'exit' to quit");
    
    let stdin = io::stdin();
    let mut reader = stdin.lock();
    let mut input = String::new();
    
    loop {
        print!("engramdb> ");
        io::stdout().flush().unwrap();
        
        input.clear();
        if reader.read_line(&mut input).unwrap() == 0 {
            break; // EOF
        }
        
        let cmd = input.trim();
        if cmd.is_empty() {
            continue;
        }
        
        let parts: Vec<&str> = cmd.split_whitespace().collect();
        
        match parts[0] {
            "exit" | "quit" => break,
            
            "help" => {
                println!("Available commands:");
                println!("  list                  - List all memories");
                println!("  view [id]             - View a specific memory");
                println!("  create                - Create a new memory (interactive)");
                println!("  edit [id]             - Edit a memory (interactive)");
                println!("  delete [id]           - Delete a memory");
                println!("  search [query]        - Search for memories");
                println!("  connect               - Add a connection (interactive)");
                println!("  disconnect            - Remove a connection (interactive)");
                println!("  exit, quit            - Exit the shell");
            },
            
            "list" => {
                if let Err(e) = list_memories(db) {
                    println!("Error: {}", e);
                }
            },
            
            "view" => {
                if parts.len() < 2 {
                    println!("Usage: view [id]");
                    continue;
                }
                
                if let Err(e) = view_memory(db, parts[1]) {
                    println!("Error: {}", e);
                }
            },
            
            "create" => {
                // Interactive create
                println!("Creating a new memory...");
                
                print!("Description: ");
                io::stdout().flush().unwrap();
                let mut description = String::new();
                reader.read_line(&mut description).unwrap();
                let description = description.trim();
                
                print!("Importance (0.0-1.0) [0.5]: ");
                io::stdout().flush().unwrap();
                let mut importance_str = String::new();
                reader.read_line(&mut importance_str).unwrap();
                let importance_str = importance_str.trim();
                let importance = if importance_str.is_empty() {
                    0.5
                } else {
                    match importance_str.parse::<f32>() {
                        Ok(val) => val,
                        Err(_) => {
                            println!("Invalid importance, using default 0.5");
                            0.5
                        }
                    }
                };
                
                print!("Tags (comma separated): ");
                io::stdout().flush().unwrap();
                let mut tags = String::new();
                reader.read_line(&mut tags).unwrap();
                let tags = tags.trim();
                
                if let Err(e) = create_memory(db, description, importance, tags) {
                    println!("Error: {}", e);
                }
            },
            
            "edit" => {
                if parts.len() < 2 {
                    println!("Usage: edit [id]");
                    continue;
                }
                
                let id = parts[1];
                
                // First display the current memory
                if let Err(e) = view_memory(db, id) {
                    println!("Error: {}", e);
                    continue;
                }
                
                // Interactive edit
                println!("Editing memory {}...", id);
                println!("(Press Enter to keep current value)");
                
                print!("Description: ");
                io::stdout().flush().unwrap();
                let mut description = String::new();
                reader.read_line(&mut description).unwrap();
                let description = description.trim();
                let description = if description.is_empty() { None } else { Some(description) };
                
                print!("Importance (0.0-1.0): ");
                io::stdout().flush().unwrap();
                let mut importance_str = String::new();
                reader.read_line(&mut importance_str).unwrap();
                let importance_str = importance_str.trim();
                let importance = if importance_str.is_empty() {
                    None
                } else {
                    match importance_str.parse::<f32>() {
                        Ok(val) => Some(val),
                        Err(_) => {
                            println!("Invalid importance, keeping current value");
                            None
                        }
                    }
                };
                
                print!("Tags (comma separated): ");
                io::stdout().flush().unwrap();
                let mut tags = String::new();
                reader.read_line(&mut tags).unwrap();
                let tags = tags.trim();
                let tags = if tags.is_empty() { None } else { Some(tags) };
                
                if let Err(e) = edit_memory(
                    db, 
                    id, 
                    description, 
                    importance, 
                    tags
                ) {
                    println!("Error: {}", e);
                }
            },
            
            "delete" => {
                if parts.len() < 2 {
                    println!("Usage: delete [id]");
                    continue;
                }
                
                if let Err(e) = delete_memory(db, parts[1], false) {
                    println!("Error: {}", e);
                }
            },
            
            "search" => {
                if parts.len() < 2 {
                    println!("Usage: search [query]");
                    continue;
                }
                
                let query = parts[1..].join(" ");
                if let Err(e) = search_memories(db, &query) {
                    println!("Error: {}", e);
                }
            },
            
            "connect" => {
                // Interactive connect
                println!("Adding a connection...");
                
                print!("Source ID: ");
                io::stdout().flush().unwrap();
                let mut source = String::new();
                reader.read_line(&mut source).unwrap();
                let source = source.trim();
                
                print!("Target ID: ");
                io::stdout().flush().unwrap();
                let mut target = String::new();
                reader.read_line(&mut target).unwrap();
                let target = target.trim();
                
                println!("Relationship type:");
                println!("  1. Association (default)");
                println!("  2. Causation");
                println!("  3. PartOf");
                println!("  4. Contains");
                println!("  5. Sequence");
                println!("  6. Custom");
                print!("Choose (1-6): ");
                io::stdout().flush().unwrap();
                
                let mut rel_choice = String::new();
                reader.read_line(&mut rel_choice).unwrap();
                let rel_choice = rel_choice.trim();
                
                let relationship = match rel_choice {
                    "2" => "Causation".to_string(),
                    "3" => "PartOf".to_string(),
                    "4" => "Contains".to_string(),
                    "5" => "Sequence".to_string(),
                    "6" => {
                        print!("Enter custom relationship type: ");
                        io::stdout().flush().unwrap();
                        let mut custom = String::new();
                        reader.read_line(&mut custom).unwrap();
                        custom.trim().to_string()
                    }
                    _ => "Association".to_string(),
                };
                
                print!("Strength (0.0-1.0) [0.8]: ");
                io::stdout().flush().unwrap();
                let mut strength_str = String::new();
                reader.read_line(&mut strength_str).unwrap();
                let strength_str = strength_str.trim();
                let strength = if strength_str.is_empty() {
                    0.8
                } else {
                    match strength_str.parse::<f32>() {
                        Ok(val) => val,
                        Err(_) => {
                            println!("Invalid strength, using default 0.8");
                            0.8
                        }
                    }
                };
                
                if let Err(e) = add_connection(db, source, target, &relationship, strength) {
                    println!("Error: {}", e);
                }
            },
            
            "disconnect" => {
                // Interactive disconnect
                println!("Removing a connection...");
                
                print!("Source ID: ");
                io::stdout().flush().unwrap();
                let mut source = String::new();
                reader.read_line(&mut source).unwrap();
                let source = source.trim();
                
                print!("Target ID: ");
                io::stdout().flush().unwrap();
                let mut target = String::new();
                reader.read_line(&mut target).unwrap();
                let target = target.trim();
                
                if let Err(e) = remove_connection(db, source, target) {
                    println!("Error: {}", e);
                }
            },
            
            _ => println!("Unknown command: {}. Type 'help' for available commands.", parts[0]),
        }
    }
    
    println!("Exiting shell.");
    Ok(())
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    // Parse command line
    let cli = Cli::parse();
    
    // Determine storage type
    let storage_type = match cli.storage_type.as_str() {
        "memory" | "mem" => StorageType::Memory,
        "single-file" | "single" => StorageType::SingleFile,
        _ => StorageType::MultiFile,
    };
    
    // Create database config
    let config = DatabaseConfig {
        storage_type: if cli.memory { StorageType::Memory } else { storage_type },
        storage_path: cli.database.map(|p| p.to_string_lossy().to_string()),
        cache_size: 100,
        vector_index_config: VectorIndexConfig::default(),
    };
    
    // Initialize the database
    let mut db = match Database::new(config) {
        Ok(mut db) => {
            let _ = db.initialize();
            db
        },
        Err(e) => {
            eprintln!("Error opening database: {}", e);
            return Err(e);
        }
    };
    
    // Handle commands based on CLI parsing
    match cli.command {
        Some(Command::List) => list_memories(&db),
        Some(Command::View { id }) => view_memory(&db, &id),
        Some(Command::Create { description, importance, tags }) => {
            create_memory(&mut db, &description, importance, &tags).map(|_| ())
        },
        Some(Command::Edit { id, description, importance, tags }) => {
            edit_memory(&mut db, &id, description.as_deref(), importance, tags.as_deref())
        },
        Some(Command::Delete { id, force }) => delete_memory(&mut db, &id, force),
        Some(Command::Search { query }) => search_memories(&db, &query),
        Some(Command::Connect { source, target, relationship, strength }) => {
            add_connection(&mut db, &source, &target, &relationship, strength)
        },
        Some(Command::Disconnect { source, target }) => {
            remove_connection(&mut db, &source, &target)
        },
        Some(Command::Shell) | None => interactive_shell(&mut db),
    }
}