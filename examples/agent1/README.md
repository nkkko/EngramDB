# EngramDB + PydanticAI Integration Examples

This directory contains examples demonstrating how to use EngramDB with PydanticAI to create powerful AI agents with memory.

## Flask Website Generator

`flask_website_generator.py` is a demonstration of building an AI agent that can design and generate a Flask website based on user requirements while maintaining context across multiple interactions.

### Key Features

- **Persistent Memory**: Uses EngramDB to store chat history, requirements, and generated code components
- **Structured Output**: Uses PydanticAI to validate and structure agent responses
- **Long-term Context**: Retrieves past interactions using vector similarity search
- **Connection Tracking**: Creates relationships between requirements and their implementing code components
- **File Generation**: Outputs a complete, runnable Flask website

### How it Works

1. The agent stores all user requirements, chat messages, and code components as memory nodes in EngramDB
2. Each memory node is vectorized for similarity search (using a simple mock implementation for demo purposes)
3. The agent creates connections between related memories (e.g., requirements and their implementing components)
4. Dynamic system prompts retrieve context from EngramDB to maintain coherence across interactions
5. Built-in validation ensures the agent generates complete, valid code components

### Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. Install EngramDB from the local repository and other dependencies:
   ```bash
   # From the examples/agent directory:
   pip install -e ../../python/  # Install EngramDB from local repo
   pip install pydantic-ai flask
   ```

3. Configure API Keys and Settings:
   Copy the example environment file and add your API keys:
   ```bash
   cp .env.example .env
   # Edit .env with your preferred text editor to add API keys
   ```

   The `.env` file allows you to configure:
   - The LLM model to use (OpenAI, Anthropic, or Groq)
   - Your API keys for the chosen provider
   - Storage locations for the EngramDB database and generated website files

### Running the Example

1. Run the example:
   ```bash
   python flask_website_generator.py
   ```

   or with web interface:
   ```bash
   python flask_website_generator.py web
   ```

3. Start by describing the website you want to build:
   ```
   You: I need a simple blog website with user authentication
   ```

4. Continue the conversation, providing more details as the agent asks for them

5. Type `save` at any time to save the generated files to disk

6. Type `exit` to quit the application

### Example Usage

```
You: I need a simple blog website with user authentication

Agent: I'll help you build a simple blog website with user authentication using Flask. Let's start with a clear plan and structure.

Generated Components:
  - main (route): The main Flask application file with basic configuration

Suggested next steps:
  - Add database models for users and blog posts
  - Create authentication routes (login, register, logout)
  - Design templates for the blog layout
```

### Why EngramDB?

This example demonstrates EngramDB's value for LLM agent implementations by providing:

1. **Persistent Memory**: Agents can maintain context across sessions
2. **Efficient Retrieval**: Vector similarity search enables semantic retrieval of past interactions
3. **Relationship Modeling**: Connections between memory nodes allow tracking relationships between requirements and implementations
4. **Structured Storage**: Memory nodes can store arbitrary attributes with type information
5. **Simple API**: Clean, idiomatic Python interface for storing and retrieving memories

This approach helps solve common challenges in LLM applications like context limitations, maintaining coherence across interactions, and retrieving relevant information from past conversations.

## Troubleshooting

### Missing Dependencies
- **Error**: `ModuleNotFoundError: No module named 'engramdb_py'`
  **Solution**: Install EngramDB from the local repository as shown in the Development Setup.

- **Error**: `ModuleNotFoundError: No module named 'pydantic_ai'`
  **Solution**: Make sure you've installed the pydantic-ai package with `pip install pydantic-ai`.

### API Key Issues
- **Error**: "API key not found" or "Authentication error"
  **Solution**: Ensure you've added your API key to the `.env` file and it's correct.

### Memory Database
- If you want to start with a fresh memory database, delete the `agent_memory.engramdb` file.
- To backup generated websites, copy the output directory specified in your `.env` file.