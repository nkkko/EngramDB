"""
Example demonstrating text embedding functionality in EngramDB.

This example shows how to:
1. Create an embedding service to convert text to vectors
2. Create memories from text content
3. Use these memories for semantic search
"""

import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import engramdb_py
sys.path.append(str(Path(os.path.dirname(__file__)).parent.parent / "python"))

try:
    from engramdb_py import Database, MemoryNode, EmbeddingService
except ImportError:
    print("Error: Could not import EngramDB Python bindings.")
    print("Please ensure you've built the Python bindings with 'embeddings' feature.")
    sys.exit(1)

def main():
    print("EngramDB Text Embeddings Example")
    print("--------------------------------")
    
    # Create a database for our examples
    db_path = Path("examples_text_embeddings.engramdb")
    db = Database.file_based(db_path)
    print(f"Database created at {db_path}")
    
    # Initialize the embedding service (will use mock embeddings if ML deps not installed)
    embedding_service = EmbeddingService.default()
    print(f"Embedding service initialized (dimensions: {embedding_service.dimensions()})")
    
    # Create memory nodes from text
    texts = [
        "Artificial intelligence is transforming how we interact with technology",
        "Machine learning algorithms can identify patterns in large datasets",
        "Natural language processing enables computers to understand human speech",
        "Computer vision systems can recognize objects in images and videos",
        "Reinforcement learning helps AI agents learn through trial and error",
        "Quantum computing may accelerate certain AI algorithms exponentially"
    ]
    
    # Store all texts as memories
    print("\nCreating memories from text...")
    memory_ids = []
    for i, text in enumerate(texts):
        # The create_memory_from_text method handles everything:
        # 1. Converts text to embeddings
        # 2. Creates a memory node with those embeddings
        # 3. Stores the text in the 'content' attribute
        # 4. Saves the memory to the database
        memory_id = db.create_memory_from_text(
            text=text,
            embedding_service=embedding_service,
            category="AI concepts",
            title=f"AI Concept {i+1}"
        )
        memory_ids.append(memory_id)
        print(f"Created memory {i+1}: ID {memory_id}")
    
    # Demonstrate semantic search
    print("\nPerforming semantic searches...")
    
    # Create query embeddings
    queries = [
        "How is AI changing our technology interactions?",
        "Finding patterns in data",
        "Understanding human language",
        "Recognizing objects in pictures",
        "Learning through experimentation",
        "Advanced computing methods"
    ]
    
    for i, query in enumerate(queries):
        print(f"\nQuery: '{query}'")
        
        # Generate embeddings for the query
        query_embedding = embedding_service.generate_for_query(query)
        
        # Search for similar memories
        results = db.search_similar(query_embedding, limit=2, threshold=0.0)
        
        # Display results
        print(f"Found {len(results)} results:")
        for j, (memory_id, similarity) in enumerate(results):
            # Load the memory to get its content
            memory = db.load(memory_id)
            content = memory.get_attribute("content")
            print(f"  Result {j+1}: (similarity: {similarity:.4f})")
            print(f"    Content: '{content}'")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()