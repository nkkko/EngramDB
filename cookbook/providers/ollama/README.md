# EngramDB with Ollama (Local LLMs)

This directory contains examples demonstrating how to use EngramDB with locally-run LLMs via Ollama.

## Examples

- `basic_integration.py`: Simple integration between EngramDB and Ollama
- `edge_optimization.py`: Optimizing for resource-constrained environments
- `local_rag.py`: Building a fully local RAG system

## Requirements

```bash
pip install ollama engramdb-py
```

## Ollama Setup

1. Install Ollama from https://ollama.com/
2. Start the Ollama service
3. Pull the models you want to use:
   ```bash
   ollama pull llama3:8b
   ```

## Running the Examples

```bash
python basic_integration.py
```