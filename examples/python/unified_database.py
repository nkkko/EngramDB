import os
import shutil
from typing import List, Dict, Any, Tuple
import engramdb_py as engramdb

def main():
    # Create a unified database with different types of memories
    db_path = "./unified_db"
    os.makedirs(db_path, exist_ok=True)
    db = engramdb.Database.file_based(db_path)
    print(f"Unified database created at: {db_path}")
    
    # Populate the database with different types of memory records
    memory_ids = populate_database(db)
    
    # Demonstrate unified search across all memory types
    unified_search(db)
    
    # Clean up
    print("\nCleaning up...")
    for memory_id in memory_ids:
        db.delete(memory_id)
    
    shutil.rmtree(db_path)
    print("Done!")

def populate_database(db: engramdb.Database) -> List[Any]:
    """Populate the database with different types of memories"""
    memory_ids = []
    
    # Add various types of memory records
    memory_ids.extend(add_notes(db))
    memory_ids.extend(add_tasks(db))
    memory_ids.extend(add_contacts(db))
    memory_ids.extend(add_documents(db))
    
    print(f"Added {len(memory_ids)} memories to the unified database")
    return memory_ids

def add_notes(db: engramdb.Database) -> List[Any]:
    """Add note-type memories to the database"""
    note_ids = []
    
    # Note 1
    note1 = engramdb.MemoryNode([0.1, 0.2, 0.8, 0.3])
    note1.set_attribute("type", "note")
    note1.set_attribute("title", "Project Ideas")
    note1.set_attribute("content", "List of potential new projects to explore next quarter")
    note1.set_attribute("tags", ["project", "planning", "ideas"])
    note_id1 = db.save(note1)
    note_ids.append(note_id1)
    
    # Note 2
    note2 = engramdb.MemoryNode([0.3, 0.1, 0.7, 0.2])
    note2.set_attribute("type", "note")
    note2.set_attribute("title", "Meeting Summary")
    note2.set_attribute("content", "Key points from the team strategy meeting")
    note2.set_attribute("tags", ["meeting", "summary", "team"])
    note_id2 = db.save(note2)
    note_ids.append(note_id2)
    
    print(f"Added {len(note_ids)} notes")
    return note_ids

def add_tasks(db: engramdb.Database) -> List[Any]:
    """Add task-type memories to the database"""
    task_ids = []
    
    # Task 1
    task1 = engramdb.MemoryNode([0.2, 0.8, 0.1, 0.3])
    task1.set_attribute("type", "task")
    task1.set_attribute("title", "Complete Documentation")
    task1.set_attribute("description", "Finish writing the API documentation for the project")
    task1.set_attribute("priority", "high")
    task1.set_attribute("status", "pending")
    task1.set_attribute("tags", ["documentation", "coding", "deadline"])
    task_id1 = db.save(task1)
    task_ids.append(task_id1)
    
    # Task 2
    task2 = engramdb.MemoryNode([0.1, 0.7, 0.2, 0.4])
    task2.set_attribute("type", "task")
    task2.set_attribute("title", "Review Pull Requests")
    task2.set_attribute("description", "Review and approve team pull requests")
    task2.set_attribute("priority", "medium")
    task2.set_attribute("status", "pending")
    task2.set_attribute("tags", ["review", "coding", "team"])
    task_id2 = db.save(task2)
    task_ids.append(task_id2)
    
    print(f"Added {len(task_ids)} tasks")
    return task_ids

def add_contacts(db: engramdb.Database) -> List[Any]:
    """Add contact-type memories to the database"""
    contact_ids = []
    
    # Contact 1
    contact1 = engramdb.MemoryNode([0.9, 0.1, 0.1, 0.2])
    contact1.set_attribute("type", "contact")
    contact1.set_attribute("name", "John Smith")
    contact1.set_attribute("email", "john@example.com")
    contact1.set_attribute("company", "Tech Solutions Inc.")
    contact1.set_attribute("role", "Developer")
    contact1.set_attribute("tags", ["colleague", "project", "developer"])
    contact_id1 = db.save(contact1)
    contact_ids.append(contact_id1)
    
    # Contact 2
    contact2 = engramdb.MemoryNode([0.8, 0.2, 0.1, 0.3])
    contact2.set_attribute("type", "contact")
    contact2.set_attribute("name", "Sarah Johnson")
    contact2.set_attribute("email", "sarah@example.com")
    contact2.set_attribute("company", "Design Studio")
    contact2.set_attribute("role", "UX Designer")
    contact2.set_attribute("tags", ["designer", "collaboration", "external"])
    contact_id2 = db.save(contact2)
    contact_ids.append(contact_id2)
    
    print(f"Added {len(contact_ids)} contacts")
    return contact_ids

def add_documents(db: engramdb.Database) -> List[Any]:
    """Add document-type memories to the database"""
    document_ids = []
    
    # Document 1
    doc1 = engramdb.MemoryNode([0.3, 0.3, 0.8, 0.1])
    doc1.set_attribute("type", "document")
    doc1.set_attribute("title", "Technical Specification")
    doc1.set_attribute("content", "Detailed technical specification for the new API")
    doc1.set_attribute("format", "markdown")
    doc1.set_attribute("tags", ["specification", "api", "documentation"])
    doc_id1 = db.save(doc1)
    document_ids.append(doc_id1)
    
    # Document 2
    doc2 = engramdb.MemoryNode([0.4, 0.2, 0.7, 0.1])
    doc2.set_attribute("type", "document")
    doc2.set_attribute("title", "Project Plan")
    doc2.set_attribute("content", "Roadmap and timeline for the project implementation")
    doc2.set_attribute("format", "pdf")
    doc2.set_attribute("tags", ["planning", "timeline", "project"])
    doc_id2 = db.save(doc2)
    document_ids.append(doc_id2)
    
    print(f"Added {len(document_ids)} documents")
    return document_ids

def unified_search(db: engramdb.Database):
    """Perform unified search across all memory types"""
    # Search for project-related memories across all types
    print("\nSearch for project-related content:")
    project_vector = [0.3, 0.2, 0.7, 0.2]  # Vector representing project-related content
    project_results = db.search_similar(project_vector, limit=3, threshold=0.0)
    
    print_search_results(db, project_results, "project-related")
    
    # Search for technical content across all types
    print("\nSearch for technical content:")
    technical_vector = [0.2, 0.3, 0.8, 0.1]  # Vector representing technical content
    technical_results = db.search_similar(technical_vector, limit=3, threshold=0.0)
    
    print_search_results(db, technical_results, "technical")
    
    # Search for people and contacts
    print("\nSearch for people and contacts:")
    people_vector = [0.8, 0.2, 0.1, 0.2]  # Vector representing people/contacts
    people_results = db.search_similar(people_vector, limit=2, threshold=0.0)
    
    print_search_results(db, people_results, "people/contacts")

def print_search_results(db: engramdb.Database, results: List[Tuple[Any, float]], search_type: str):
    """Print search results in a formatted way"""
    print(f"Found {len(results)} {search_type} results:")
    
    for memory_id, similarity in results:
        memory = db.load(memory_id)
        memory_type = memory.get_attribute("type")
        
        # Format output based on memory type
        if memory_type == "note":
            title = memory.get_attribute("title")
            print(f"  - Note: {title} (similarity: {similarity:.4f})")
            print(f"    Content: {memory.get_attribute('content')}")
            
        elif memory_type == "task":
            title = memory.get_attribute("title")
            priority = memory.get_attribute("priority")
            print(f"  - Task: {title} (similarity: {similarity:.4f})")
            print(f"    Priority: {priority}, Description: {memory.get_attribute('description')}")
            
        elif memory_type == "contact":
            name = memory.get_attribute("name")
            company = memory.get_attribute("company")
            print(f"  - Contact: {name} (similarity: {similarity:.4f})")
            print(f"    Company: {company}, Role: {memory.get_attribute('role')}")
            
        elif memory_type == "document":
            title = memory.get_attribute("title")
            format = memory.get_attribute("format")
            print(f"  - Document: {title} (similarity: {similarity:.4f})")
            print(f"    Format: {format}")
            
        else:
            print(f"  - Unknown type: {memory_type} (similarity: {similarity:.4f})")
        
        # Print tags if available
        if memory.has_attribute("tags"):
            tags = memory.get_attribute("tags")
            print(f"    Tags: {', '.join(tags)}")

if __name__ == "__main__":
    main()