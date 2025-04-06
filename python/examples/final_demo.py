"""
EngramDB Example: AI Coding Agent for Bug Fixing

This example demonstrates how an AI coding agent would use EngramDB
to store and retrieve memories while fixing bugs in a codebase.
"""
import sys
import os

# Add the parent directory to sys.path so Python can find the engramdb package
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from engramdb_py import MemoryNode, Database, AttributeFilter
import numpy as np

def run_example():
    try:
        # Create an in-memory database
        print("Creating in-memory database...")
        db = Database.in_memory()
        
        print("\nCreating memories for AI Coding Agent bug fixing workflow...")
        
        # Create a simple bug fixing workflow with connected memories
        
        # Memory 1: Bug Report
        bug_report = MemoryNode(np.array([0.82, 0.41, 0.33, 0.25], dtype=np.float32))
        bug_report.set_attribute("memory_type", "bug_report")
        bug_report.set_attribute("title", "NullPointerException in UserService.java")
        bug_report.set_attribute("content", "NullPointerException at line 45")
        bug_report_id = db.save(bug_report)
        
        # Memory 2: Bug Analysis
        analysis = MemoryNode(np.array([0.85, 0.44, 0.30, 0.22], dtype=np.float32))
        analysis.set_attribute("memory_type", "analysis_result")
        analysis.set_attribute("title", "Root Cause Analysis")
        analysis.set_attribute("content", "userRepository is null")
        analysis_id = db.save(analysis)
        
        # Memory 3: Solution Implementation
        implementation = MemoryNode(np.array([0.76, 0.48, 0.37, 0.31], dtype=np.float32))
        implementation.set_attribute("memory_type", "code_snippet")
        implementation.set_attribute("title", "Constructor Implementation")
        implementation.set_attribute("content", "public UserService(UserRepository repo) { this.userRepository = repo; }")
        implementation_id = db.save(implementation)
        
        # Memory 4: Verification
        verification = MemoryNode(np.array([0.72, 0.45, 0.33, 0.35], dtype=np.float32))
        verification.set_attribute("memory_type", "testing_outcome")
        verification.set_attribute("title", "Fix Verification")
        verification.set_attribute("success", True)
        verification_id = db.save(verification)
        
        # Connect these memories to form a workflow
        bug_report = db.load(bug_report_id)
        bug_report.add_connection(analysis_id, "led_to")
        db.save(bug_report)
        
        analysis = db.load(analysis_id)
        analysis.add_connection(implementation_id, "led_to")
        db.save(analysis)
        
        implementation = db.load(implementation_id)
        implementation.add_connection(verification_id, "verified_by")
        db.save(implementation)
        
        # Demonstrate vector similarity search
        print("\nSearching for similar bug reports...")
        new_bug_vector = np.array([0.80, 0.43, 0.32, 0.28], dtype=np.float32)
        results = db.search_similar(new_bug_vector, limit=2, threshold=0.0)
        
        for memory_id, similarity in results:
            memory = db.load(memory_id)
            print(f"  {memory.get_attribute('title')} ({memory.get_attribute('memory_type')}) - Similarity: {similarity:.4f}")
        
        # Demonstrate attribute filtering
        print("\nFinding all code snippets:")
        type_filter = AttributeFilter.equals("memory_type", "code_snippet")
        query_results = db.query().with_attribute_filter(type_filter).execute()
        
        for memory in query_results:
            print(f"  {memory.get_attribute('title')}")
            print(f"    Content: {memory.get_attribute('content')}")
        
        # Demonstrate connection traversal
        print("\nTracing the bug fixing workflow:")
        current_memory = db.load(bug_report_id)
        print(f"Step 1: {current_memory.get_attribute('title')}")
        
        step = 2
        while True:
            connections = current_memory.get_connections()
            if not connections:
                break
                
            next_id = connections[0][0]
            next_memory = db.load(next_id)
            print(f"Step {step}: {next_memory.get_attribute('title')}")
            
            current_memory = next_memory
            step += 1
        
        print("\nEngramDB AI Coding Agent example complete!")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_example()