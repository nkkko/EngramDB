import os
import shutil
from typing import Dict, List, Any, Tuple
import engramdb_py as engramdb

def main():
    # Create a database for our memory graph
    storage_dir = "./memory_graph_db"
    os.makedirs(storage_dir, exist_ok=True)
    db = engramdb.Database.file_based(storage_dir)
    print(f"Memory graph database created at: {storage_dir}")
    
    # Create a knowledge graph of connected memories
    person_ids, concept_ids, event_ids = create_knowledge_graph(db)
    
    # Query connections and relationships
    print("\nQuerying knowledge graph...")
    query_knowledge_graph(db, person_ids, concept_ids, event_ids)
    
    # Clean up
    print("\nCleaning up...")
    all_ids = db.list_all()
    for memory_id in all_ids:
        db.delete(memory_id)
    
    shutil.rmtree(storage_dir)
    print("Done!")

def create_knowledge_graph(db: engramdb.Database) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    Create a knowledge graph with three types of nodes:
    - People
    - Concepts
    - Events
    
    Each node type is connected to others with relationships.
    """
    # Create Person nodes
    person_ids = []
    alice = engramdb.MemoryNode([0.1, 0.2, 0.3, 0.4])
    alice.set_attribute("type", "person")
    alice.set_attribute("name", "Alice")
    alice.set_attribute("occupation", "Researcher")
    alice_id = db.save(alice)
    person_ids.append(alice_id)
    
    bob = engramdb.MemoryNode([0.2, 0.3, 0.4, 0.5])
    bob.set_attribute("type", "person")
    bob.set_attribute("name", "Bob")
    bob.set_attribute("occupation", "Engineer")
    bob_id = db.save(bob)
    person_ids.append(bob_id)
    
    # Create Concept nodes
    concept_ids = []
    ai = engramdb.MemoryNode([0.5, 0.5, 0.5, 0.5])
    ai.set_attribute("type", "concept")
    ai.set_attribute("name", "Artificial Intelligence")
    ai.set_attribute("field", "Computer Science")
    ai_id = db.save(ai)
    concept_ids.append(ai_id)
    
    ml = engramdb.MemoryNode([0.6, 0.6, 0.6, 0.6])
    ml.set_attribute("type", "concept")
    ml.set_attribute("name", "Machine Learning")
    ml.set_attribute("field", "Computer Science")
    ml_id = db.save(ml)
    concept_ids.append(ml_id)
    
    # Create Event nodes
    event_ids = []
    conference = engramdb.MemoryNode([0.7, 0.7, 0.7, 0.7])
    conference.set_attribute("type", "event")
    conference.set_attribute("name", "AI Conference 2025")
    conference.set_attribute("date", "2025-04-15")
    conference_id = db.save(conference)
    event_ids.append(conference_id)
    
    # Create relationships
    # Alice knows Bob
    alice = db.load(alice_id)
    alice.set_relationship("knows", bob_id)
    db.save(alice)
    
    # Bob knows Alice
    bob = db.load(bob_id)
    bob.set_relationship("knows", alice_id)
    db.save(bob)
    
    # Alice is expert in AI
    alice = db.load(alice_id)
    alice.set_relationship("expert_in", ai_id)
    db.save(alice)
    
    # Bob is expert in ML
    bob = db.load(bob_id)
    bob.set_relationship("expert_in", ml_id)
    db.save(bob)
    
    # AI includes ML
    ai = db.load(ai_id)
    ai.set_relationship("includes", ml_id)
    db.save(ai)
    
    # Conference is about AI
    conference = db.load(conference_id)
    conference.set_relationship("about", ai_id)
    conference.set_relationship("about", ml_id)
    db.save(conference)
    
    # Alice attended Conference
    alice = db.load(alice_id)
    alice.set_relationship("attended", conference_id)
    db.save(alice)
    
    # Bob attended Conference
    bob = db.load(bob_id)
    bob.set_relationship("attended", conference_id)
    db.save(bob)
    
    print(f"Created knowledge graph with {len(person_ids)} people, {len(concept_ids)} concepts, and {len(event_ids)} events")
    return person_ids, concept_ids, event_ids

def query_knowledge_graph(db: engramdb.Database, person_ids: List[Any], concept_ids: List[Any], event_ids: List[Any]):
    """Query the knowledge graph to extract relationships"""
    # Who knows whom
    print("\nSocial connections:")
    for person_id in person_ids:
        person = db.load(person_id)
        name = person.get_attribute("name")
        
        if person.has_relationships("knows"):
            knows_ids = person.get_relationships("knows")
            for knows_id in knows_ids:
                known_person = db.load(knows_id)
                known_name = known_person.get_attribute("name")
                print(f"  {name} knows {known_name}")
    
    # Expertise relationships
    print("\nExpertise:")
    for person_id in person_ids:
        person = db.load(person_id)
        name = person.get_attribute("name")
        
        if person.has_relationships("expert_in"):
            expert_ids = person.get_relationships("expert_in")
            for expert_id in expert_ids:
                concept = db.load(expert_id)
                concept_name = concept.get_attribute("name")
                print(f"  {name} is an expert in {concept_name}")
    
    # Event attendance
    print("\nEvent attendance:")
    for event_id in event_ids:
        event = db.load(event_id)
        event_name = event.get_attribute("name")
        
        # Find all people who attended this event by searching through person nodes
        for person_id in person_ids:
            person = db.load(person_id)
            if person.has_relationships("attended"):
                attended_ids = person.get_relationships("attended")
                if event_id in attended_ids:
                    person_name = person.get_attribute("name")
                    print(f"  {person_name} attended {event_name}")
        
        # What was the event about
        if event.has_relationships("about"):
            about_ids = event.get_relationships("about")
            about_concepts = []
            for about_id in about_ids:
                concept = db.load(about_id)
                about_concepts.append(concept.get_attribute("name"))
            
            print(f"  {event_name} was about: {', '.join(about_concepts)}")

if __name__ == "__main__":
    main()