# Agent Memory Database MVP

## Core Components

1. **Core memory representation**
   - Implement basic `MemoryNode` struct with essential fields
   - Simple vector embedding storage and retrieval
   - Basic relationship graph between nodes

2. **Minimal storage engine**
   - File-based persistence of memory nodes
   - Simple CRUD operations
   - No advanced tiering yet

3. **Basic vector search**
   - Implement simplified vector similarity search
   - Single-algorithm approach without adaptive features
   - Focus on correctness over optimization

4. **Simple query API**
   - Create a small, focused query interface
   - Support for finding similar memories
   - Basic temporal filtering

5. **Single-agent focus**
   - Skip multi-user/multi-agent complexities
   - Avoid advanced concurrency challenges
   - Run in single-threaded mode initially

This MVP could be completed in 2-3 months with 1-2 engineers while validating the core value proposition of the specialized agent memory database.