use engramdb::{Database, MemoryNode, Result};

/// Loads a sample dataset with an AI Chat Assistant memory database
/// that demonstrates how EngramDB can be used to create persistent memory
/// for language models.
/// 
/// # Arguments
///
/// * `db` - A mutable reference to an EngramDB database
///
/// # Returns
///
/// A Result containing a vector of IDs for the created nodes
pub fn load_sample_dataset(db: &mut Database) -> Result<Vec<uuid::Uuid>> {
    let mut node_ids = Vec::new();
    
    // Clear any existing memories first
    match db.clear_all() {
        Ok(_) => (),
        Err(e) => eprintln!("Warning: Could not clear database: {}", e),
    }
    
    //---------------------- USER INFORMATION ----------------------//
    
    // Memory 1: User Profile
    let mut user_profile = MemoryNode::new(vec![0.82, 0.41, 0.33, 0.25, 0.1]);
    user_profile.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("User Profile: Alice Chen".to_string()));
    user_profile.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("user_profile".to_string()));
    user_profile.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.95));
    user_profile.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Name: Alice Chen\nOccupation: Software Engineer\nInterests: Machine Learning, Hiking, Photography\nPreferred Coding Languages: Python, Rust\nLearning Goals: Improve knowledge of distributed systems".to_string()
        ));
    user_profile.set_attribute("memory_type".to_string(), 
        engramdb::core::AttributeValue::String("factual".to_string()));
    user_profile.set_attribute("timestamp".to_string(), 
        engramdb::core::AttributeValue::String("2025-03-15T10:30:00".to_string()));
    let user_profile_id = db.save(&user_profile)?;
    node_ids.push(user_profile_id);
    
    // Memory 2: User Preference
    let mut user_preference = MemoryNode::new(vec![0.75, 0.52, 0.38, 0.31, 0.15]);
    user_preference.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Communication Preferences".to_string()));
    user_preference.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("user_preference".to_string()));
    user_preference.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.85));
    user_preference.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Alice prefers concise explanations with code examples. She appreciates technical depth but wants explanations to be practical. She values efficiency in conversations and prefers direct answers with minimal preamble.".to_string()
        ));
    user_preference.set_attribute("memory_type".to_string(), 
        engramdb::core::AttributeValue::String("factual".to_string()));
    user_preference.set_attribute("timestamp".to_string(), 
        engramdb::core::AttributeValue::String("2025-03-15T10:45:00".to_string()));
    let user_preference_id = db.save(&user_preference)?;
    node_ids.push(user_preference_id);
    
    //---------------------- PAST CONVERSATIONS ----------------------//
    
    // Memory 3: Previous Conversation - Rust Concurrency
    let mut conversation1 = MemoryNode::new(vec![0.85, 0.44, 0.30, 0.22, 0.18]);
    conversation1.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Conversation About Rust Concurrency".to_string()));
    conversation1.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("conversation".to_string()));
    conversation1.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.75));
    conversation1.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Alice asked about Rust's concurrency model and how it differs from Go's. We discussed Rust's ownership model and how it prevents data races at compile time, versus Go's goroutines and channels approach. Alice was especially interested in the Send and Sync traits.".to_string()
        ));
    conversation1.set_attribute("memory_type".to_string(), 
        engramdb::core::AttributeValue::String("interaction".to_string()));
    conversation1.set_attribute("topic".to_string(), 
        engramdb::core::AttributeValue::String("rust_concurrency".to_string()));
    conversation1.set_attribute("timestamp".to_string(), 
        engramdb::core::AttributeValue::String("2025-03-18T14:30:00".to_string()));
    let conversation1_id = db.save(&conversation1)?;
    node_ids.push(conversation1_id);
    
    // Memory 4: Previous Conversation - ML Project Advice
    let mut conversation2 = MemoryNode::new(vec![0.79, 0.51, 0.34, 0.28, 0.22]);
    conversation2.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("ML Project Architecture Advice".to_string()));
    conversation2.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("conversation".to_string()));
    conversation2.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.8));
    conversation2.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Alice described her machine learning project for predicting software defects. She needed advice on structuring her pipeline for real-time predictions. We discussed using a feature store to manage training and inference features, and setting up a monitoring system to detect model drift. Alice decided to use MLflow for experiment tracking and model registry.".to_string()
        ));
    conversation2.set_attribute("memory_type".to_string(), 
        engramdb::core::AttributeValue::String("interaction".to_string()));
    conversation2.set_attribute("topic".to_string(), 
        engramdb::core::AttributeValue::String("machine_learning".to_string()));
    conversation2.set_attribute("timestamp".to_string(), 
        engramdb::core::AttributeValue::String("2025-03-25T09:45:00".to_string()));
    let conversation2_id = db.save(&conversation2)?;
    node_ids.push(conversation2_id);
    
    //---------------------- TECHNICAL KNOWLEDGE ----------------------//
    
    // Memory 5: Technical Knowledge - Distributed Systems
    let mut knowledge1 = MemoryNode::new(vec![0.76, 0.48, 0.37, 0.31, 0.25]);
    knowledge1.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Distributed Systems Design Principles".to_string()));
    knowledge1.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("knowledge".to_string()));
    knowledge1.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.9));
    knowledge1.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Key principles of distributed systems that Alice is learning about: 1) CAP Theorem (consistency, availability, partition tolerance), 2) Eventual consistency models, 3) Consensus algorithms like Raft and Paxos, 4) Idempotency and exactly-once delivery semantics, 5) Event sourcing and CQRS patterns.".to_string()
        ));
    knowledge1.set_attribute("memory_type".to_string(), 
        engramdb::core::AttributeValue::String("factual".to_string()));
    knowledge1.set_attribute("topic".to_string(), 
        engramdb::core::AttributeValue::String("distributed_systems".to_string()));
    knowledge1.set_attribute("timestamp".to_string(), 
        engramdb::core::AttributeValue::String("2025-04-01T11:00:00".to_string()));
    let knowledge1_id = db.save(&knowledge1)?;
    node_ids.push(knowledge1_id);
    
    // Memory 6: Technical Resource
    let mut resource1 = MemoryNode::new(vec![0.72, 0.45, 0.33, 0.35, 0.28]);
    resource1.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Recommended Book: Designing Data-Intensive Applications".to_string()));
    resource1.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("resource".to_string()));
    resource1.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.85));
    resource1.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Recommended 'Designing Data-Intensive Applications' by Martin Kleppmann for Alice's distributed systems learning. The book covers data models, storage engines, encoding, replication, partitioning, consistency, consensus, batch and stream processing.".to_string()
        ));
    resource1.set_attribute("memory_type".to_string(), 
        engramdb::core::AttributeValue::String("factual".to_string()));
    resource1.set_attribute("topic".to_string(), 
        engramdb::core::AttributeValue::String("distributed_systems".to_string()));
    resource1.set_attribute("resource_type".to_string(), 
        engramdb::core::AttributeValue::String("book".to_string()));
    resource1.set_attribute("timestamp".to_string(), 
        engramdb::core::AttributeValue::String("2025-04-01T11:15:00".to_string()));
    resource1.set_attribute("url".to_string(), 
        engramdb::core::AttributeValue::String("https://dataintensive.net/".to_string()));
    let resource1_id = db.save(&resource1)?;
    node_ids.push(resource1_id);
    
    // Memory 7: Current Project Context
    let mut current_project = MemoryNode::new(vec![0.86, 0.41, 0.32, 0.24, 0.19]);
    current_project.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Current Project: Distributed ML Feature Store".to_string()));
    current_project.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("project".to_string()));
    current_project.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.95));
    current_project.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Alice is building a distributed feature store for ML features with these requirements: 1) Low-latency access for online inference, 2) Consistent view of features for training and inference, 3) Support for point-in-time correctness, 4) Integration with multiple data sources, 5) Feature versioning and lineage tracking.".to_string()
        ));
    current_project.set_attribute("memory_type".to_string(), 
        engramdb::core::AttributeValue::String("factual".to_string()));
    current_project.set_attribute("topic".to_string(), 
        engramdb::core::AttributeValue::String("feature_store".to_string()));
    current_project.set_attribute("timestamp".to_string(), 
        engramdb::core::AttributeValue::String("2025-04-02T10:00:00".to_string()));
    let current_project_id = db.save(&current_project)?;
    node_ids.push(current_project_id);
    
    // Create meaningful connections between our nodes
    
    // User information connections
    db.connect(user_profile_id, user_preference_id, "Contains".to_string(), 0.9)?;
    
    // Connect conversations to user profile
    db.connect(conversation1_id, user_profile_id, "RefersTo".to_string(), 0.9)?;
    db.connect(conversation2_id, user_profile_id, "RefersTo".to_string(), 0.9)?;
    
    // Connect knowledge to relevant conversations
    db.connect(conversation1_id, knowledge1_id, "RelatedTo".to_string(), 0.7)?;
    db.connect(conversation2_id, current_project_id, "Precedes".to_string(), 0.85)?;
    
    // Connect resources to knowledge areas
    db.connect(resource1_id, knowledge1_id, "Supports".to_string(), 0.95)?;
    
    // Connect current project to previous conversations and knowledge
    db.connect(current_project_id, knowledge1_id, "RequiresKnowledgeOf".to_string(), 0.9)?;
    db.connect(current_project_id, conversation2_id, "EvolvesFrom".to_string(), 0.85)?;
    
    // Connect user profile to interests
    db.connect(user_profile_id, knowledge1_id, "InterestedIn".to_string(), 0.85)?;
    db.connect(user_profile_id, current_project_id, "WorkingOn".to_string(), 0.95)?;
    
    // Timeline connections
    db.connect(conversation1_id, conversation2_id, "Precedes".to_string(), 0.8)?; // Time-based connection
    db.connect(conversation2_id, current_project_id, "Precedes".to_string(), 0.8)?; // Time-based connection
    
    Ok(node_ids)
}

/// Loads a reduced version of the sample dataset with a simplified AI assistant memory,
/// optimized for demonstrating the core features of EngramDB.
///
/// # Arguments
///
/// * `db` - A mutable reference to an EngramDB database
///
/// # Returns
///
/// A Result containing a vector of IDs for the created nodes
pub fn load_minimal_dataset(db: &mut Database) -> Result<Vec<uuid::Uuid>> {
    let mut node_ids = Vec::new();
    
    // Clear any existing memories first
    match db.clear_all() {
        Ok(_) => (),
        Err(e) => eprintln!("Warning: Could not clear database: {}", e),
    }
    
    // Create nodes representing a simplified AI assistant memory
    
    // User profile node
    let mut user_profile = MemoryNode::new(vec![0.9, 0.1, 0.2]);
    user_profile.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("User Profile: Alice Chen".to_string()));
    user_profile.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("user_profile".to_string()));
    user_profile.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.9));
    user_profile.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Name: Alice Chen\nOccupation: Software Engineer\nInterests: Machine Learning, Distributed Systems".to_string()
        ));
    let user_id = db.save(&user_profile)?;
    node_ids.push(user_id);
    
    // Conversation memory node
    let mut conversation = MemoryNode::new(vec![0.8, 0.7, 0.3]);
    conversation.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Previous Conversation on Distributed Systems".to_string()));
    conversation.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("conversation".to_string()));
    conversation.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.8));
    conversation.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Alice asked about distributed consensus algorithms and their trade-offs in production systems."
        ));
    let conversation_id = db.save(&conversation)?;
    node_ids.push(conversation_id);
    
    // Knowledge node
    let mut knowledge = MemoryNode::new(vec![0.6, 0.8, 0.4]);
    knowledge.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Distributed Systems Principles".to_string()));
    knowledge.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("knowledge".to_string()));
    knowledge.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.7));
    knowledge.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Core principles of distributed systems including CAP theorem, consistency models, and consensus algorithms."
        ));
    let knowledge_id = db.save(&knowledge)?;
    node_ids.push(knowledge_id);
    
    // Resource node
    let mut resource = MemoryNode::new(vec![0.5, 0.7, 0.8]);
    resource.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Recommended Resource: DDIA Book".to_string()));
    resource.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("resource".to_string()));
    resource.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.8));
    resource.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Designing Data-Intensive Applications by Martin Kleppmann - an excellent book on distributed systems principles."
        ));
    let resource_id = db.save(&resource)?;
    node_ids.push(resource_id);
    
    // Project node
    let mut project = MemoryNode::new(vec![0.3, 0.4, 0.9]);
    project.set_attribute("title".to_string(), 
        engramdb::core::AttributeValue::String("Current Project: Feature Store".to_string()));
    project.set_attribute("category".to_string(), 
        engramdb::core::AttributeValue::String("project".to_string()));
    project.set_attribute("importance".to_string(), 
        engramdb::core::AttributeValue::Float(0.9));
    project.set_attribute("content".to_string(), 
        engramdb::core::AttributeValue::String(
            "Alice is building a distributed feature store for machine learning features."
        ));
    let project_id = db.save(&project)?;
    node_ids.push(project_id);
    
    // Create meaningful connections between nodes
    
    // User connections
    db.connect(user_id, conversation_id, "Participated".to_string(), 0.9)?;
    db.connect(user_id, project_id, "WorkingOn".to_string(), 0.9)?;
    db.connect(user_id, knowledge_id, "InterestedIn".to_string(), 0.8)?;
    
    // Knowledge and resources
    db.connect(knowledge_id, resource_id, "ReferencedBy".to_string(), 0.9)?;
    db.connect(conversation_id, knowledge_id, "Discussed".to_string(), 0.9)?;
    
    // Project connections
    db.connect(project_id, knowledge_id, "RequiresKnowledgeOf".to_string(), 0.8)?;
    db.connect(conversation_id, project_id, "Informs".to_string(), 0.7)?;
    
    Ok(node_ids)
}