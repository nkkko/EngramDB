#!/usr/bin/env python3
"""
EngramDB LLMs.txt Generator

This script generates an llms.txt file for the EngramDB python package,
which can then be converted to XML context using llms_txt2ctx.
"""

import os
import sys
from pathlib import Path
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate llms.txt file for EngramDB python package")
    parser.add_argument("--output", "-o", default="llms-engramdb.txt",
                        help="Output file path (default: llms-engramdb.txt)")
    parser.add_argument("--full", "-f", action="store_true",
                        help="Generate llms-full.txt with more detailed documentation")
    args = parser.parse_args()

    # Project root directory
    root_dir = Path(os.path.dirname(os.path.abspath(__file__))).parent
    
    # Read README content
    readme_path = root_dir / "README.md"
    readme_content = readme_path.read_text()
    
    # Extract title and summary
    title = "EngramDB Python"
    summary = "EngramDB Python bindings for the specialized agent memory database system"
    
    # Basic information
    info_text = """
EngramDB is a specialized database designed for agent memory management. 
It provides vector search, attribute-based filtering, and memory connections
in a unified API. Use it to build systems that can effectively store and
retrieve contextual information, particularly for AI agents.

When using EngramDB, remember to:
- Use vectors consistently (same dimensions)
- Store structured attributes for faster filtering
- Connect related memories using relationship types
- Consider thread-safe operations for multi-agent systems
"""

    # Build sections
    docs_section = """
## Docs

- [README](python/README.md): Python bindings overview and basic usage
- [Basic Usage Example](python/examples/basic_usage.py): Full example of managing an AI agent's memories
- [Sample Data Usage](python/examples/sample_data_usage.py): How to use the sample data generator
- [Thread Safety](python/examples/thread_safe_example.py): Thread-safe operations for multi-agent systems
"""

    examples_section = """
## Examples

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
"""

    optional_section = """
## Optional

- [Contributing Guide](CONTRIBUTING.md): How to contribute to EngramDB
- [Thread Safety Guide](python/docs/thread_safety.md): Detailed guide for thread-safe operations
- [Advanced Queries](python/docs/advanced_queries.md): Complex query examples and patterns
"""

    # Combine all sections
    llms_txt = f"""# {title}

> {summary}

{info_text}

{docs_section}

{examples_section}

{optional_section}
"""

    # Write to output file
    output_path = Path(args.output)
    output_path.write_text(llms_txt)
    print(f"Generated {output_path}")
    
    # If full flag is set, create a more detailed version
    if args.full:
        full_output_path = output_path.parent / "llms-full-engramdb.txt"
        
        # Extract code examples from README
        code_examples = re.findall(r'```python\n(.*?)```', readme_content, re.DOTALL)
        
        # Read example file
        example_path = root_dir / "examples" / "basic_usage.py"
        example_content = example_path.read_text()
        
        # Create expanded examples section
        expanded_examples = examples_section + "\n\n### Full Example\n\n```python\n" + example_content + "\n```\n"
        
        # Create full version with more content
        full_llms_txt = f"""# {title}

> {summary}

{info_text}

## API Overview

The EngramDB Python module provides these main components:

- Database: In-memory or file-based storage for memory nodes
- MemoryNode: Container for vector embeddings and attributes
- ThreadSafeDatabase: Thread-safe version for multi-agent systems
- RelationshipType: Enum for common memory relationships

{docs_section}

{expanded_examples}

{optional_section}
"""
        
        full_output_path.write_text(full_llms_txt)
        print(f"Generated {full_output_path}")

if __name__ == "__main__":
    main()