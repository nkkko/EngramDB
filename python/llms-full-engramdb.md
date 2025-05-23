<project title="EngramDB Python" summary='EngramDB Python bindings for the specialized agent memory database system'>

# EngramDB Python

> EngramDB Python bindings for the specialized agent memory database system


EngramDB is a specialized database designed for agent memory management. 
It provides vector search, attribute-based filtering, and memory connections
in a unified API. Use it to build systems that can effectively store and
retrieve contextual information, particularly for AI agents.

When using EngramDB, remember to:
- Use vectors consistently (same dimensions)
- Store structured attributes for faster filtering
- Connect related memories using relationship types
- Consider thread-safe operations for multi-agent systems

<section title="API Overview">
The EngramDB Python module provides these main components:

- Database: In-memory or file-based storage for memory nodes
- MemoryNode: Container for vector embeddings and attributes
- ThreadSafeDatabase: Thread-safe version for multi-agent systems
- RelationshipType: Enum for common memory relationships
</section>

<docs>

<doc title="README" url="python/README.md">Python bindings overview and basic usage</doc>

<doc title="Basic Usage Example" url="python/examples/basic_usage.py">Full example of managing an AI agent's memories</doc>

<doc title="Sample Data Usage" url="python/examples/sample_data_usage.py">How to use the sample data generator</doc>

<doc title="Thread Safety" url="python/examples/thread_safe_example.py">Thread-safe operations for multi-agent systems</doc>

</docs>

<examples>
```python
# Create an in-memory database
db = engramdb.Database.in_memory()

# Create a memory node with vector embeddings
memory = engramdb.MemoryNode([0.1, 0.2, 0.3, 0.4])
memory.set_attribute("title", "Important information")
memory.set_attribute("importance", 0.8)

# Save to database
memory_id = db.save(memory)
print(f"Saved memory with ID: {memory_id}")

# Search for similar memories
results = db.search_similar([0.15, 0.25, 0.35, 0.45], limit=5, threshold=0.7)
for memory_id, score in results:
    memory = db.load(memory_id)
    print(f"Found similar memory: {memory.get_attribute('title')} (score: {score:.2f})")

# Create connections between memories
db.connect(memory_id1, memory_id2, "related_to", 0.9)

# Query with filters
attribute_filter = engramdb.AttributeFilter.greater_than("importance", 0.7)
results = db.query().with_attribute_filter(attribute_filter).execute()
```
</examples>

<section title="# Full Example">
```python
"""
EngramDB Example: AI Coding Agent for Bug Fixing

This example demonstrates how an AI coding agent would use EngramDB
to store and retrieve memories while fixing bugs in a codebase.
"""
import engramdb
import numpy as np
import uuid

def run_example():
    # Create an in-memory database
    print("Creating in-memory database...")
    db = engramdb.Database.in_memory()
    
    print("\nCreating memories for AI Coding Agent bug fixing workflow...")
    
    # Memory 1: Bug Report
    bug_report_vector = np.array([0.82, 0.41, 0.33, 0.25], dtype=np.float32)
    bug_report = engramdb.MemoryNode(bug_report_vector)
    bug_report.set_attribute("memory_type", "bug_report")
    bug_report.set_attribute("title", "NullPointerException in UserService.java")
    bug_report.set_attribute("content", "NullPointerException in UserService.java at line 45 when calling getUserDetails()")
    bug_report.set_attribute("severity", 0.8)
    bug_report.set_attribute("module", "user-management")
    bug_report_id = db.save(bug_report)
    print(f"Created memory: Bug Report (ID: {bug_report_id})")
    
    # Memory 2: Codebase Structure
    codebase_vector = np.array([0.74, 0.52, 0.28, 0.31], dtype=np.float32)
    codebase_memory = engramdb.MemoryNode(codebase_vector)
    codebase_memory.set_attribute("memory_type", "codebase_structure")
    codebase_memory.set_attribute("title", "UserService Dependencies")
    codebase_memory.set_attribute("content", "Traversed UserService.java, identified dependency on UserRepository.java")
    codebase_memory.set_attribute("module", "user-management")
    codebase_memory.set_attribute("related_files", ["UserService.java", "UserRepository.java"])
    codebase_id = db.save(codebase_memory)
    print(f"Created memory: Codebase Structure (ID: {codebase_id})")
    
    # Memory 3: Bug Analysis
    analysis_vector = np.array([0.85, 0.44, 0.30, 0.22], dtype=np.float32)
    analysis_memory = engramdb.MemoryNode(analysis_vector)
    analysis_memory.set_attribute("memory_type", "analysis_result")
    analysis_memory.set_attribute("title", "Root Cause Analysis")
    analysis_memory.set_attribute("content", "userRepository is null leading to NullPointerException in getUserDetails()")
    analysis_memory.set_attribute("confidence", 0.95)
    analysis_memory.set_attribute("module", "user-management")
    analysis_id = db.save(analysis_memory)
    print(f"Created memory: Bug Analysis (ID: {analysis_id})")
    
    # Memory 4: Solution Planning
    solution_vector = np.array([0.79, 0.51, 0.34, 0.28], dtype=np.float32)
    solution_memory = engramdb.MemoryNode(solution_vector)
    solution_memory.set_attribute("memory_type", "proposed_fix")
    solution_memory.set_attribute("title", "Initialization Fix")
    solution_memory.set_attribute("content", "Initialize userRepository in the constructor of UserService.java")
    solution_memory.set_attribute("estimated_success", 0.9)
    solution_memory.set_attribute("module", "user-management")
    solution_id = db.save(solution_memory)
    print(f"Created memory: Solution Planning (ID: {solution_id})")
    
    # Memory 5: Implementation
    implementation_vector = np.array([0.76, 0.48, 0.37, 0.31], dtype=np.float32)
    implementation_memory = engramdb.MemoryNode(implementation_vector)
    implementation_memory.set_attribute("memory_type", "code_snippet")
    implementation_memory.set_attribute("title", "Constructor Implementation")
    implementation_memory.set_attribute("content", """
    public UserService(UserRepository userRepository) {
        this.userRepository = userRepository;
    }
    """)
    implementation_memory.set_attribute("file", "UserService.java")
    implementation_memory.set_attribute("module", "user-management")
    implementation_id = db.save(implementation_memory)
    print(f"Created memory: Implementation (ID: {implementation_id})")
    
    # Memory 6: Verification
    verification_vector = np.array([0.72, 0.45, 0.33, 0.35], dtype=np.float32)
    verification_memory = engramdb.MemoryNode(verification_vector)
    verification_memory.set_attribute("memory_type", "testing_outcome")
    verification_memory.set_attribute("title", "Fix Verification")
    verification_memory.set_attribute("content", "Unit test passed. No NullPointerException when calling getUserDetails()")
    verification_memory.set_attribute("success", True)
    verification_memory.set_attribute("module", "user-management")
    verification_id = db.save(verification_memory)
    print(f"Created memory: Verification (ID: {verification_id})")
    
    # Create connections between memories to establish the workflow
    print("\nEstablishing connections between memories...")
    
    # Connect bug report to codebase structure
    bug_report = db.load(bug_report_id)
    bug_report.add_connection(codebase_id, "led_to")
    db.save(bug_report)
    
    # Connect codebase structure to analysis
    codebase_memory = db.load(codebase_id)
    codebase_memory.add_connection(analysis_id, "led_to")
    db.save(codebase_memory)
    
    # Connect analysis to solution
    analysis_memory = db.load(analysis_id)
    analysis_memory.add_connection(solution_id, "led_to")
    db.save(analysis_memory)
    
    # Connect solution to implementation
    solution_memory = db.load(solution_id)
    solution_memory.add_connection(implementation_id, "implemented_as")
    db.save(solution_memory)
    
    # Connect implementation to verification
    implementation_memory = db.load(implementation_id)
    implementation_memory.add_connection(verification_id, "verified_by")
    db.save(implementation_memory)
    
    # Also connect bug report directly to verification for the complete fix chain
    bug_report = db.load(bug_report_id)
    bug_report.add_connection(verification_id, "resolved_by")
    db.save(bug_report)
    
    print("Memory connections established.")
    
    # List all memories
    all_ids = db.list_all()
    print(f"\nDatabase contains {len(all_ids)} memories")
    
    # Demo 1: Vector similarity search for similar bugs
    print("\nDemo 1: Searching for similar bug reports...")
    
    # This vector represents a new bug with similar characteristics
    new_bug_vector = np.array([0.80, 0.43, 0.32, 0.28], dtype=np.float32)
    
    results = db.search_similar(new_bug_vector, limit=3, threshold=0.0)
    print(f"Found {len(results)} similar bug-related memories:")
    
    for memory_id, similarity in results:
        memory = db.load(memory_id)
        print(f"  {memory.get_attribute('title')} ({memory.get_attribute('memory_type')}) - Similarity: {similarity:.4f}")
    
    # Demo 2: Filter queries to find specific memory types
    print("\nDemo 2: Finding all implementation memories in the user-management module...")
    
    type_filter = engramdb.AttributeFilter.equals("memory_type", "code_snippet")
    module_filter = engramdb.AttributeFilter.equals("module", "user-management")
    
    query_results = db.query()\
        .with_attribute_filter(type_filter)\
        .with_attribute_filter(module_filter)\
        .execute()
    
    print(f"Found {len(query_results)} code_snippet memories in user-management module:")
    for memory in query_results:
        print(f"  {memory.get_attribute('title')}")
        print(f"    File: {memory.get_attribute('file')}")
        print(f"    Content: {memory.get_attribute('content').strip()}")
    
    # Demo 3: Follow connections to trace the bug fixing process
    print("\nDemo 3: Tracing the full bug fixing workflow...")
    
    # Start with the bug report
    current_memory = db.load(bug_report_id)
    print(f"Starting with: {current_memory.get_attribute('title')} ({current_memory.get_attribute('memory_type')})")
    
    # Follow "led_to" and other connections to trace the workflow
    traced_memories = []
    
    while True:
        # Get all outgoing connections
        connections = current_memory.get_connections()
        
        # If no more connections, we're done
        if not connections:
            break
            
        # Just follow the first connection for this simple example
        # (In a real system, you'd have more complex traversal logic)
        next_id = connections[0][0]  # (id, type) pairs
        next_memory = db.load(next_id)
        
        traced_memories.append((next_memory.get_attribute('memory_type'), next_memory.get_attribute('title')))
        
        # Move to the next memory
        current_memory = next_memory
    
    print("Bug fixing workflow trace:")
    for i, (memory_type, title) in enumerate(traced_memories, 1):
        print(f"  Step {i}: [{memory_type}] {title}")
    
    print("\nEngramDB AI Coding Agent example complete!")

if __name__ == "__main__":
    run_example()
```
</section>

<section title="Optional">
- [Contributing Guide](CONTRIBUTING.md): How to contribute to EngramDB
- [Thread Safety Guide](python/docs/thread_safety.md): Detailed guide for thread-safe operations
- [Advanced Queries](python/docs/advanced_queries.md): Complex query examples and patterns
</section>

</project>
