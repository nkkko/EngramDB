"""
EngramDB: Engram Database

A specialized database system designed for agent memory management.
"""

try:
    from ._engramdb import *
    print("Imported _engramdb successfully")
except ImportError as e:
    import sys
    print(f"Error importing _engramdb: {e}", file=sys.stderr)
    # For debugging purposes only
    def sum_as_string(a, b):
        return str(a + b)
    
    class MemoryNode:
        def __init__(self, embeddings):
            raise NotImplementedError("EngramDB module not properly installed")
    
    class Database:
        @staticmethod
        def in_memory():
            raise NotImplementedError("EngramDB module not properly installed")
        
        @staticmethod
        def file_based(path):
            raise NotImplementedError("EngramDB module not properly installed")
    
    class RelationshipType:
        Association = 0
        Causation = 1
        Sequence = 2
        Contains = 3

__version__ = "0.1.0"