---
marp: true
theme: default
paginate: true
backgroundColor: "#FFFFFF"
header: "**EngramDB**"
footer: "© 2025 EngramDB"
style: |
  .columns {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 1rem;
  }
  img[alt~="center"] {
    display: block;
    margin: 0 auto;
  }
  section {
    font-size: 28px;
  }
  h1 {
    font-size: 42px;
    color: #333;
  }
  h2 {
    font-size: 36px;
    color: #0066cc;
  }
---

<!-- _class: lead -->
# **EngramDB**
## A purpose-built graph vector database for AI agent memory

![bg right:40% 90%](/Users/nikola/dev/engramdb/assets/logo/light.svg)

---

# The Problem

- **Current agentic AI systems lack effective memory mechanisms**
- Vector databases are not optimized for agent use cases
- LLM applications need efficient knowledge storage and retrieval
- Existing solutions don't balance performance with developer ergonomics

---
# Why Agent Memory Matters

- **Limited Context Window**: Agents only "see" 10-20K tokens of context at any time
- **Discontinuous Perception**: As Barry Zhang from Anthropic notes, agents operate with "lethal phases" between actions
- **Memory Persistence**: EngramDB bridges these gaps with durable, contextual memory
- **Budget Constraints**: Memory retrieval reduces token usage vs. repeated context stuffing
- **Error Recovery**: Persistent memory enables agents to resume after failures

---
# Thinking Like Your Agent

<div class="columns">
<div>

### Agent Challenges
- Limited to what's in context window
- "Closing eyes" between actions
- No persistent understanding
- Expensive context repetition
- Difficulty with complex state tracking

</div>
<div>

### EngramDB Solutions
- Rich temporal memory layers
- Connection-based reasoning
- Budget-aware memory retrieval
- Efficient state persistence
- Tool execution history storage

</div>
</div>

> "Everything that the model knows about the current state of the world is going to be explained in that 10 to 20k tokens." - Barry Zhang
---

# Introducing EngramDB

> A lightweight, embedded graph vector database inspired by SQLite's design principles and implemented in Rust.

- **Purpose-built** for AI agent memory storage and retrieval
- **Self-contained** with no external dependencies
- **Developer-friendly** API and SDK for multiple languages
- **Performance-optimized** for AI workloads

---

![bg 70%](/Users/nikola/dev/engramdb/assets/engram_web_graph.jpg)

---

# Core Technical Features

<div class="columns">
<div>

### Current Capabilities
- In-memory & file-based storage
- Flexible data model with attributes
- Connection-based graph support
- Python SDK with native bindings
- Basic vector search

</div>
<div>

### Coming Soon
- Advanced vector indexing (HNSW)
- Transaction support (ACID)
- Query optimization
- Connection pooling
- Multi-client concurrency

</div>
</div>

---

# Technical Architecture

![bg right:65% 95%](/Users/nikola/dev/engramdb/assets/engram_arch_2.svg)

---

![bg 75%](/Users/nikola/dev/engramdb/assets/engramdb_arch.svg)


---

# Vector Indexing Capabilities

- **HNSW Algorithm** for fast approximate nearest neighbor search
- **Hybrid indexing** combining exact and approximate methods
- **Customizable similarity functions** for domain-specific use cases
- **Memory-efficient storage** through product quantization
- **Scalable** to handle millions of embeddings

```rust
// Configurable indexing
db.configure_index()
   .vector_algorithm(VectorAlgorithm::HNSW)
   .with_parameters(HNSWParams {
       m: 16,                 // Maximum connections per node
       ef_construction: 100,  // Size of dynamic candidate list
       ef: 10,                // Size of dynamic list during search
   })
   .build()?;
```

---

# Engineered for AI Memory

- **Engrams**: Rich data units with vectors, metadata, and connections
- **Temporal layers**: Track memory evolution over time
- **Custom query language** for complex memory retrieval
- **Connection-based reasoning** using graph relationships

```
# Example query language:
FIND MEMORIES
SIMILAR TO [0.1, 0.3, 0.5, 0.2] WITH SIMILARITY > 0.7
WHERE attribute.category = "meeting" AND attribute.importance > 0.8
CREATED WITHIN LAST 7 DAYS
CONNECTED TO "d8f7a2e5-1c3b-4a6d-9e8f-5b7a2c3d1e0f" BY "Association"
LIMIT 10
```

---

# Agent-Friendly API

<div class="columns">
<div>

### Rust Example
```rust
// Create a memory node with embedding
let memory = MemoryNode::new(vec![0.1, 0.2, 0.3]);
memory.set_attribute("type", "conversation");
memory.set_attribute("importance", 0.8);

// Save to database
db.save(&memory)?;

// Retrieve similar memories
let results = db.query()
    .similar_to(&memory.vector, 0.7)
    .with_attribute("type", "conversation")
    .limit(5)
    .execute()?;
```

</div>
<div>

### Python Example
```python
# Create a memory node with embedding
memory = MemoryNode([0.1, 0.2, 0.3])
memory.set_attribute("type", "conversation")
memory.set_attribute("importance", 0.8)

# Save to database
db.save(memory)

# Retrieve similar memories
results = db.query()\
    .similar_to(memory.vector, 0.7)\
    .with_attribute("type", "conversation")\
    .limit(5)\
    .execute()
```

</div>
</div>

---

# Implementation Roadmap

<div class="columns">
<div>

### Phase 1 (Short-term)
- HNSW algorithm implementation
- Read/write concurrency
- Pydantic.ai agent prototype
- Transaction support
- Hybrid vector indexing
- Query language for memory retrieval

</div>
<div>

### Phase 2 & 3 (Mid/Long-term)
- Advanced query optimization
- Connection pooling
- Schema migration tools
- Self-contained packaging
- Cross-language clients
- Benchmarking suite

</div>
</div>

---

# Competitive Advantages

<div class="columns">
<div>

### EngramDB
- Lightweight embedded database
- Purpose-built for AI memory
- Graph-based connections
- Developer experience focus
- Self-contained design
- Fast prototyping

</div>
<div>

### Alternatives
- Chroma/Pinecone: vector-only
- LanceDB: Limited connection support
- Neo4j: No native vector support
- SQLite+pgvector: Performance tradeoffs
- Weaviate: Complex deployment
- Postgres: High resource overhead

</div>
</div>

---
### EngramDB for Agents
- **Context Window Maximizer**: Only retrieve what's needed, when needed
- **Budget-Aware**: Optimize token usage through targeted memory retrieval
- **Agent Learning**: Track performance to improve over time
- **Tool-Memory Integration**: Store tool execution history and results
- **Error Recovery**: Robustness through memory persistence
---
# Supporting Multi-Agent Systems

<div class="columns">
<div>

### The Future Is Multi-Agent
- Anthropic predicts multi-agent systems in production by end of 2025
- Agents need shared memory to collaborate
- Communication across contexts requires persistence
- Separation of concerns demands role-specific memories

</div>
<div>

### EngramDB Multi-Agent Features
- Shared memory pools with access controls
- Agent-specific memory partitions
- Cross-agent memory connections
- Memory-based communication channels
- Role and permission management

</div>
</div>

---

# Business Model & Market Strategy

- **Open-core model** for sustainable development
- **Focus on AX** best agent experience
- **Ultimate DevEx** for AI developers
- **EngramDB Cloud** for managed hosting (future)
- **Enterprise support** for organizations
- **Focus markets**: AI startups, research labs, enterprise AI teams
- **Initial target**: AI agent developers needing memory solutions

---

# Team

- **Core Development**: Rust & database engineering experts
- **AI Integration**: Machine learning & LLM specialists
- **Operations**: Scaling & infrastructure veterans
- **Business**: Experience in developer tools & B2B SaaS

*Actively growing our team with passionate engineers and AI researchers*

---

# Investment Opportunity

- Seeking pre-seed funding to validate the concept
- Proven track record in devtools
- Capital allocation:
  - 50% Engineering team expansion
  - 45% Marketing/developer relations
  - 5% Operations

- Clear path to market with growing adoption metrics
- Strategic partnerships in development

---

# Call to Action

- **Investors**: Join us in shaping how AI agents remember
- **Developers**: Try our alpha release and provide feedback
- **Partners**: Explore integration opportunities
- **Team**: We're hiring founding engineers

**Contact**: info@engramdb.com | www.engramdb.com

![bg right:30% 80%](/Users/nikola/dev/engramdb/assets/logo/dark.svg)

---

# Thank You!

**Contact**: nikola@engramdb.com