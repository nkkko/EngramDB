import os
import uuid
import json
import shutil
from typing import List, Dict, Any, Optional
from datetime import datetime
import engramdb_py as engramdb

class AgentMemory:
    """Simple agent memory management using EngramDB"""
    
    def __init__(self, storage_path: Optional[str] = None):
        """
        Initialize the agent memory system
        
        Args:
            storage_path: Path to store memories. If None, uses in-memory storage.
        """
        if storage_path:
            # Ensure directory exists
            os.makedirs(os.path.dirname(storage_path) if os.path.dirname(storage_path) else '.', exist_ok=True)
            self.db = engramdb.Database.file_based(storage_path)
            self.storage_path = storage_path
            print(f"Agent memory initialized with file storage at: {storage_path}")
        else:
            self.db = engramdb.Database.in_memory()
            self.storage_path = None
            print("Agent memory initialized with in-memory storage")
        
        # Track session ID
        self.session_id = str(uuid.uuid4())
        print(f"Session ID: {self.session_id}")
    
    def store_message(self, role: str, content: str) -> str:
        """
        Store a conversation message
        
        Args:
            role: Message role (e.g., 'user', 'assistant', 'system')
            content: Message content
            
        Returns:
            ID of the stored memory
        """
        vector = self._text_to_vector(content)
        node = engramdb.MemoryNode(vector)
        
        # Set attributes
        node.set_attribute("type", "message")
        node.set_attribute("role", role)
        node.set_attribute("content", content)
        node.set_attribute("timestamp", datetime.now().isoformat())
        node.set_attribute("session_id", self.session_id)
        
        # Save to database
        memory_id = self.db.save(node)
        return str(memory_id)
    
    def store_fact(self, fact: str, source: Optional[str] = None) -> str:
        """
        Store a fact or piece of information
        
        Args:
            fact: The fact to store
            source: Optional source of the fact
            
        Returns:
            ID of the stored memory
        """
        vector = self._text_to_vector(fact)
        node = engramdb.MemoryNode(vector)
        
        # Set attributes
        node.set_attribute("type", "fact")
        node.set_attribute("content", fact)
        if source:
            node.set_attribute("source", source)
        node.set_attribute("timestamp", datetime.now().isoformat())
        node.set_attribute("session_id", self.session_id)
        
        # Save to database
        memory_id = self.db.save(node)
        return str(memory_id)
    
    def store_task(self, title: str, description: str, status: str = "pending") -> str:
        """
        Store a task
        
        Args:
            title: Task title
            description: Task description
            status: Task status (e.g., 'pending', 'completed')
            
        Returns:
            ID of the stored memory
        """
        vector = self._text_to_vector(f"{title} {description}")
        node = engramdb.MemoryNode(vector)
        
        # Set attributes
        node.set_attribute("type", "task")
        node.set_attribute("title", title)
        node.set_attribute("description", description)
        node.set_attribute("status", status)
        node.set_attribute("timestamp", datetime.now().isoformat())
        node.set_attribute("session_id", self.session_id)
        
        # Save to database
        memory_id = self.db.save(node)
        return str(memory_id)
    
    def update_task_status(self, task_id: str, new_status: str) -> bool:
        """
        Update a task's status
        
        Args:
            task_id: ID of the task to update
            new_status: New status for the task
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Convert string ID to UUID
            uuid_id = uuid.UUID(task_id)
            
            # Load the task
            node = self.db.load(uuid_id)
            
            # Check if it's a task
            if node.get_attribute("type") != "task":
                print(f"Memory with ID {task_id} is not a task")
                return False
            
            # Update the status
            node.set_attribute("status", new_status)
            node.set_attribute("updated_at", datetime.now().isoformat())
            
            # Save back to database
            self.db.save(node)
            return True
            
        except Exception as e:
            print(f"Error updating task status: {e}")
            return False
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Search for memories similar to the query
        
        Args:
            query: Search query
            limit: Maximum number of results to return
            
        Returns:
            List of similar memories with metadata
        """
        vector = self._text_to_vector(query)
        results = self.db.search_similar(vector, limit=limit, threshold=0.0)
        
        formatted_results = []
        for memory_id, similarity in results:
            try:
                node = self.db.load(memory_id)
                
                # Basic memory information
                memory_data = {
                    "id": str(memory_id),
                    "type": node.get_attribute("type"),
                    "timestamp": node.get_attribute("timestamp"),
                    "similarity": similarity
                }
                
                # Add type-specific fields
                memory_type = node.get_attribute("type")
                if memory_type == "message":
                    memory_data["role"] = node.get_attribute("role")
                    memory_data["content"] = node.get_attribute("content")
                elif memory_type == "fact":
                    memory_data["content"] = node.get_attribute("content")
                    if node.has_attribute("source"):
                        memory_data["source"] = node.get_attribute("source")
                elif memory_type == "task":
                    memory_data["title"] = node.get_attribute("title")
                    memory_data["description"] = node.get_attribute("description")
                    memory_data["status"] = node.get_attribute("status")
                
                formatted_results.append(memory_data)
            except Exception as e:
                print(f"Error processing memory {memory_id}: {e}")
        
        return formatted_results
    
    def get_recent_messages(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent conversation messages
        
        Args:
            limit: Maximum number of messages to return
            
        Returns:
            List of recent messages
        """
        all_ids = self.db.list_all()
        messages = []
        
        for memory_id in all_ids:
            try:
                node = self.db.load(memory_id)
                
                # Check if it's a message from this session
                if (node.get_attribute("type") == "message" and 
                    node.get_attribute("session_id") == self.session_id):
                    
                    messages.append({
                        "id": str(memory_id),
                        "role": node.get_attribute("role"),
                        "content": node.get_attribute("content"),
                        "timestamp": node.get_attribute("timestamp")
                    })
            except Exception as e:
                print(f"Error processing message {memory_id}: {e}")
        
        # Sort by timestamp
        messages.sort(key=lambda x: x["timestamp"])
        
        # Return the most recent messages
        return messages[-limit:] if len(messages) > limit else messages
    
    def get_pending_tasks(self) -> List[Dict[str, Any]]:
        """
        Get all pending tasks
        
        Returns:
            List of pending tasks
        """
        all_ids = self.db.list_all()
        tasks = []
        
        for memory_id in all_ids:
            try:
                node = self.db.load(memory_id)
                
                # Check if it's a pending task
                if (node.get_attribute("type") == "task" and 
                    node.get_attribute("status") == "pending"):
                    
                    tasks.append({
                        "id": str(memory_id),
                        "title": node.get_attribute("title"),
                        "description": node.get_attribute("description"),
                        "timestamp": node.get_attribute("timestamp")
                    })
            except Exception as e:
                print(f"Error processing task {memory_id}: {e}")
        
        # Sort by timestamp (oldest first)
        tasks.sort(key=lambda x: x["timestamp"])
        
        return tasks
    
    def clear_session(self) -> int:
        """
        Clear all memories from the current session
        
        Returns:
            Number of memories deleted
        """
        all_ids = self.db.list_all()
        deleted_count = 0
        
        for memory_id in all_ids:
            try:
                node = self.db.load(memory_id)
                
                # Check if it's from this session
                if node.get_attribute("session_id") == self.session_id:
                    self.db.delete(memory_id)
                    deleted_count += 1
            except Exception as e:
                print(f"Error deleting memory {memory_id}: {e}")
        
        return deleted_count
    
    def cleanup(self) -> None:
        """Clean up resources and delete storage file if using file storage"""
        if self.storage_path and os.path.exists(self.storage_path):
            try:
                # Get the directory containing the storage file
                dir_path = os.path.dirname(self.storage_path)
                if not dir_path:
                    dir_path = self.storage_path  # If storage_path is just a filename
                
                # Remove the directory and all its contents
                if os.path.isdir(dir_path):
                    shutil.rmtree(dir_path)
                else:
                    os.remove(self.storage_path)
                print(f"Removed storage at {self.storage_path}")
            except Exception as e:
                print(f"Error cleaning up storage: {e}")
    
    def _text_to_vector(self, text: str) -> List[float]:
        """
        Convert text to a vector representation
        
        This is a simple hash-based function for demonstration purposes.
        In a real application, you would use a proper embedding model.
        
        Args:
            text: Text to convert to vector
            
        Returns:
            Vector representation of the text
        """
        import hashlib
        import numpy as np
        
        # Create a hash of the text
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        # Convert hash to a seed for numpy random
        seed = int(text_hash, 16) % (2**32 - 1)
        np.random.seed(seed)
        
        # Generate a random vector (10 dimensions)
        return np.random.random(10).tolist()


def main():
    """Demonstrate agent memory functionality"""
    # Initialize agent memory with file storage
    memory_path = "./agent_memory_demo"
    agent_memory = AgentMemory(memory_path)
    
    # Simulate a conversation
    print("\nSimulating a conversation...")
    agent_memory.store_message("system", "You are a helpful assistant.")
    agent_memory.store_message("user", "I need to plan a trip to Japan next month.")
    agent_memory.store_message("assistant", "I'd be happy to help you plan your trip to Japan. What specific aspects are you interested in?")
    agent_memory.store_message("user", "I'm interested in visiting Tokyo and Kyoto. I want to focus on cultural experiences.")
    agent_memory.store_message("assistant", "Great choices! Tokyo and Kyoto offer amazing cultural experiences.")
    
    # Store some facts
    print("\nStoring facts about Japan...")
    agent_memory.store_fact("Tokyo is the capital city of Japan", "general knowledge")
    agent_memory.store_fact("Kyoto was the former capital of Japan for more than 1000 years", "general knowledge")
    agent_memory.store_fact("The bullet train (Shinkansen) connects Tokyo and Kyoto in about 2.5 hours", "travel information")
    
    # Store some tasks
    print("\nCreating tasks for trip planning...")
    task1_id = agent_memory.store_task(
        "Research Tokyo attractions",
        "Find top cultural attractions in Tokyo including museums and temples"
    )
    task2_id = agent_memory.store_task(
        "Check Kyoto accommodations",
        "Look for traditional ryokan accommodations in Kyoto"
    )
    task3_id = agent_memory.store_task(
        "Check flight options",
        "Research flights to Tokyo from user's location"
    )
    
    # Complete a task
    print("\nCompleting a task...")
    agent_memory.update_task_status(task1_id, "completed")
    
    # Retrieve recent messages
    print("\nRetrieving recent conversation:")
    recent_messages = agent_memory.get_recent_messages(5)
    for msg in recent_messages:
        print(f"[{msg['role']}]: {msg['content']}")
    
    # Retrieve pending tasks
    print("\nPending tasks:")
    pending_tasks = agent_memory.get_pending_tasks()
    for task in pending_tasks:
        print(f"- {task['title']}: {task['description']}")
    
    # Search for memories about Kyoto
    print("\nSearching for memories about Kyoto:")
    kyoto_memories = agent_memory.search_similar("Kyoto")
    for memory in kyoto_memories:
        memory_type = memory["type"]
        if memory_type == "message":
            print(f"- Message [{memory['role']}]: {memory['content']}")
        elif memory_type == "fact":
            print(f"- Fact: {memory['content']}")
        elif memory_type == "task":
            print(f"- Task: {memory['title']} - {memory['description']}")
    
    # Clean up
    print("\nCleaning up...")
    deleted_count = agent_memory.clear_session()
    print(f"Deleted {deleted_count} memories from the current session")
    agent_memory.cleanup()
    print("Done!")

if __name__ == "__main__":
    main()