# Flask Website Generator Agent

This example demonstrates a sophisticated agent that builds Flask websites based on user requirements, leveraging EngramDB for persistent memory and context management.

## Features

- **Conversation Memory**: Uses EngramDB to store chat history, requirements, and generated code
- **Contextual Understanding**: Remembers past interactions and maintains coherence across sessions
- **Code Generation**: Creates production-ready Flask application components
- **Vector Search**: Finds relevant past information using similarity search
- **Relationship Tracking**: Connects requirements to their implementing code components

## Setup

1. **Create a virtual environment**:
   ```bash
   python -m venv env
   source env/bin/activate  # On Windows: env\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -e /path/to/engramdb/python/  # Install EngramDB
   pip install pydantic-ai flask
   ```

3. **Configure the `.env` file**:
   Edit the `.env` file and add your API keys for the LLM provider you want to use.

   ```
   # Example for using Groq
   PYDANTIC_AI_MODEL="groq:llama-3.1-8b-instant"
   GROQ_API_KEY="your-api-key-here"
   ```

## Running the Agent

Start the agent:
```bash
python flask_website_generator.py
```

Or with web interface only:
```bash
python flask_website_generator.py web
```

## Example Conversation

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

## Available Commands

- `save`: Saves all generated files to disk and registers dynamic routes
- `exit`: Quits the application

## New Feature: Dynamic Website Preview

Your generated website is automatically served at the `/generated` path when you run the agent. 

- Access your running application at: http://127.0.0.1:8080/generated
- This eliminates the need to manually run the generated Flask application
- The generated files are still saved to disk for reference or deployment

When you generate new components or modify existing ones, the dynamic routes are automatically updated.

## How It Works

1. The agent uses EngramDB to store:
   - User requirements
   - Generated code components
   - Conversation history

2. Each interaction is processed through:
   - The LLM agent generates responses and code
   - Important information is stored in EngramDB nodes
   - Relationships between entities are tracked

3. When generating code, the agent:
   - Retrieves context from previous interactions
   - Creates structured components that work together
   - Validates generated code for correctness

4. Files can be generated at any time using the `save` command, creating:
   - Flask application files (app.py)
   - HTML templates
   - CSS/JS static files
   - Models and forms
   - requirements.txt

## Technical Details

- Uses EngramDB's single file storage for persistence
- Implements a simple vector embedding strategy for similarity search
- Leverages the powerful PydanticAI framework for structured agent outputs
- Components are modular and work together coherently