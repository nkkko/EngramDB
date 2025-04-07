# EngramDB with Anthropic Claude

This directory contains examples demonstrating how to use EngramDB with Anthropic's Claude models.

## Examples

- `basic_integration.py`: Simple integration between EngramDB and Claude
- `claude_rag.py`: Building a RAG system with Claude and EngramDB
- `tool_use_memory.py`: Claude with tool use and persistent memory
- `message_threading.py`: Managing Claude conversations with EngramDB

## Requirements

```bash
pip install anthropic engramdb-py
```

## API Key Setup

You'll need an Anthropic API key to run these examples. Set it as an environment variable:

```bash
export ANTHROPIC_API_KEY=your_api_key_here
```

## Running the Examples

```bash
python basic_integration.py
```