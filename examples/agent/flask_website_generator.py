"""
EngramDB + PydanticAI: Flask Website Generator Agent

This example demonstrates how to build a ChatGPT-like agent that can:
1. Design a Flask website based on user requirements
2. Use EngramDB to store chat context and generated code
3. Persist memory between interactions with the user
4. Leverage vector similarity search to recall relevant past interactions

Usage:
    python flask_website_generator.py

Dependencies:
    - pydantic-ai
    - engramdb
    - flask
"""

import os
import re
import json
import uuid
import asyncio
import tempfile
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Union

import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
import engramdb_py as engramdb

# Configuration
ENGRAMDB_PATH = "agent_memory.engramdb"
WEBSITE_OUTPUT_PATH = os.path.join(tempfile.gettempdir(), "generated_flask_website")
DEFAULT_MODEL = os.environ.get("PYDANTIC_AI_MODEL", "openai:gpt-4o")


class WebsiteComponent(BaseModel):
    """A component of the Flask website being built"""
    name: str = Field(description="Name of the component (e.g., 'login_page', 'database_model')")
    type: str = Field(description="Type of component (e.g., 'route', 'template', 'model', 'form')")
    code: str = Field(description="The complete code for this component")
    description: str = Field(description="Brief description of what this component does")


class AgentResponse(BaseModel):
    """The structured response from the website generator agent"""
    message: str = Field(description="Message to display to the user")
    generated_components: List[WebsiteComponent] = Field(
        default_factory=list,
        description="Website components generated in this interaction (if any)"
    )
    should_save_files: bool = Field(
        default=False,
        description="Whether the agent wants to save generated files to disk"
    )
    next_steps: List[str] = Field(
        default_factory=list,
        description="Suggested next steps or questions for the user"
    )


class SearchQuery(BaseModel):
    """A query for searching memory"""
    query: str = Field(description="The search query text")
    limit: int = Field(description="Maximum number of results to return")


@dataclass
class AgentContext:
    """The context object that will be injected into agent tools"""
    db: engramdb.Database
    current_chat_id: uuid.UUID = field(default_factory=uuid.uuid4)
    user_requirements: List[Dict[str, Any]] = field(default_factory=list)
    components: Dict[str, WebsiteComponent] = field(default_factory=dict)

    def text_to_vector(self, text: str) -> List[float]:
        """Convert text to a simple vector embedding

        Note: In a real application, you would use a proper embedding model here.
        This is just a demonstration using a very simplistic approach.
        """
        # For demo purposes, create a simple fixed-dimension vector
        text_bytes = text.encode('utf-8')
        # Make sure the hash value is positive and within numpy's seed range (0 to 2^32-1)
        hash_val = abs(hash(text_bytes)) % (2**32 - 1)
        np.random.seed(hash_val)
        return np.random.random(10).astype(np.float32).tolist()


# Create the agent with EngramDB dependency injection
website_generator = Agent(
    DEFAULT_MODEL,
    deps_type=AgentContext,
    result_type=AgentResponse,
    system_prompt="""
    You are an expert Flask website generator assistant. Your job is to help users build Flask
    web applications by designing and generating code based on their requirements.

    Be concise, practical, and realistic in your responses. Generate production-ready Flask code with
    proper structure, error handling, and best practices.

    You have access to a database of past interactions, which you should use to maintain context.
    When generating code, make sure it's compatible with any previously generated components.

    Each component you generate should be standalone and complete, not just snippets.

    IMPORTANT:
    - Maintain a consistent style across code components
    - Remember that users can't see your internal tools or reasoning, explain things clearly
    - Always suggest next steps to guide the user in completing their website
    - Store all important information in memory using the provided tools
    """
)


@website_generator.system_prompt
async def check_for_previous_requirements(ctx: RunContext[AgentContext]) -> str:
    """Checks EngramDB for existing project requirements"""
    try:
        # List all memory nodes
        memory_ids = ctx.deps.db.list_all()
        requirements_nodes = []

        # Look for requirement memories
        for memory_id in memory_ids:
            node = ctx.deps.db.load(memory_id)
            if node.get_attribute("memory_type") == "requirement":
                requirements_nodes.append(node)

        if not requirements_nodes:
            return "This is a new project, no existing requirements found."

        # Format requirements for the system prompt
        requirements_text = "Previous project requirements found:\n"
        for i, node in enumerate(requirements_nodes, 1):
            requirement = node.get_attribute("content")
            timestamp = node.get_attribute("timestamp")
            requirements_text += f"{i}. [{timestamp}] {requirement}\n"

        return requirements_text
    except Exception as e:
        return f"Error retrieving previous requirements: {str(e)}"


@website_generator.system_prompt
async def list_existing_components(ctx: RunContext[AgentContext]) -> str:
    """Lists any previously generated website components"""
    # Get component nodes from EngramDB
    memory_ids = ctx.deps.db.list_all()
    component_nodes = []

    for memory_id in memory_ids:
        node = ctx.deps.db.load(memory_id)
        if node.get_attribute("memory_type") == "component":
            component_nodes.append(node)

    if not component_nodes:
        return "No website components have been generated yet."

    # Format components for the system prompt
    components_text = "Previously generated components:\n"
    for i, node in enumerate(component_nodes, 1):
        component_name = node.get_attribute("name")
        component_type = node.get_attribute("type")
        component_description = node.get_attribute("description")
        components_text += f"{i}. {component_name} ({component_type}): {component_description}\n"

    return components_text


@website_generator.tool
async def store_requirement(ctx: RunContext[AgentContext], requirement: str) -> str:
    """
    Store a user requirement in EngramDB

    Args:
        requirement: The requirement text to store

    Returns:
        Message confirming the requirement was stored
    """
    # Create a memory node for the requirement
    vector = ctx.deps.text_to_vector(requirement)
    node = engramdb.MemoryNode(vector)

    # Add attributes
    node.set_attribute("memory_type", "requirement")
    node.set_attribute("content", requirement)
    node.set_attribute("timestamp", datetime.now().isoformat())
    node.set_attribute("chat_id", str(ctx.deps.current_chat_id))

    # Save to database
    requirement_id = ctx.deps.db.save(node)

    # Add to current context
    ctx.deps.user_requirements.append({
        "id": str(requirement_id),
        "content": requirement,
        "timestamp": datetime.now().isoformat()
    })

    return f"Requirement stored with ID: {requirement_id}"


@website_generator.tool
async def store_component(
    ctx: RunContext[AgentContext],
    name: str,
    component_type: str,
    code: str,
    description: str
) -> str:
    """
    Store a generated website component in EngramDB

    Args:
        name: Name of the component (e.g., 'login_page', 'user_model')
        component_type: Type of component (e.g., 'route', 'template', 'model')
        code: The complete code for this component
        description: Brief description of what this component does

    Returns:
        Message confirming the component was stored
    """
    # Create a memory node for the component
    vector = ctx.deps.text_to_vector(f"{name} {description} {code}")
    node = engramdb.MemoryNode(vector)

    # Add attributes
    node.set_attribute("memory_type", "component")
    node.set_attribute("name", name)
    node.set_attribute("type", component_type)
    node.set_attribute("code", code)
    node.set_attribute("description", description)
    node.set_attribute("timestamp", datetime.now().isoformat())
    node.set_attribute("chat_id", str(ctx.deps.current_chat_id))

    # Save to database
    component_id = ctx.deps.db.save(node)

    # Add to current context
    component = WebsiteComponent(
        name=name,
        type=component_type,
        code=code,
        description=description
    )
    ctx.deps.components[name] = component

    # Connect to relevant requirements
    for req in ctx.deps.user_requirements:
        req_id = uuid.UUID(req["id"])
        ctx.deps.db.connect(req_id, component_id, "fulfilled_by", 0.8)

    return f"Component '{name}' stored with ID: {component_id}"


@website_generator.tool
async def search_memory(
    ctx: RunContext[AgentContext],
    query: str,
    limit: int = 5
) -> List[Dict[str, Any]]:
    """
    Search EngramDB for relevant memories

    Args:
        query: The search text
        limit: Maximum number of results to return

    Returns:
        List of matching memories with their similarity scores
    """
    # Convert query to vector
    query_vector = ctx.deps.text_to_vector(query)

    # Search for similar memories
    results = ctx.deps.db.search_similar(query_vector, limit=limit, threshold=0.0)

    # Format results
    formatted_results = []
    for memory_id, similarity in results:
        node = ctx.deps.db.load(memory_id)
        memory_type = node.get_attribute("memory_type")

        # Format based on memory type
        if memory_type == "requirement":
            formatted_results.append({
                "id": str(memory_id),
                "type": "requirement",
                "content": node.get_attribute("content"),
                "timestamp": node.get_attribute("timestamp"),
                "similarity": float(similarity)
            })
        elif memory_type == "component":
            formatted_results.append({
                "id": str(memory_id),
                "type": "component",
                "name": node.get_attribute("name"),
                "component_type": node.get_attribute("type"),
                "description": node.get_attribute("description"),
                "code": node.get_attribute("code"),
                "timestamp": node.get_attribute("timestamp"),
                "similarity": float(similarity)
            })
        elif memory_type == "chat_message":
            formatted_results.append({
                "id": str(memory_id),
                "type": "chat_message",
                "role": node.get_attribute("role"),
                "content": node.get_attribute("content"),
                "timestamp": node.get_attribute("timestamp"),
                "similarity": float(similarity)
            })

    return formatted_results


@website_generator.tool
async def store_chat_message(
    ctx: RunContext[AgentContext],
    role: str,
    content: str
) -> str:
    """
    Store a chat message in EngramDB

    Args:
        role: The role of the message sender ('user' or 'assistant')
        content: The message content

    Returns:
        Message confirming the chat message was stored
    """
    # Create a memory node for the message
    vector = ctx.deps.text_to_vector(content)
    node = engramdb.MemoryNode(vector)

    # Add attributes
    node.set_attribute("memory_type", "chat_message")
    node.set_attribute("role", role)
    node.set_attribute("content", content)
    node.set_attribute("timestamp", datetime.now().isoformat())
    node.set_attribute("chat_id", str(ctx.deps.current_chat_id))

    # Save to database
    message_id = ctx.deps.db.save(node)

    return f"Chat message stored with ID: {message_id}"


# Note: The save_files function has been removed and its functionality
# moved directly into the places where it was called, to avoid issues with RunContext


@website_generator.tool
async def get_existing_components(ctx: RunContext[AgentContext]) -> Dict[str, Dict[str, Any]]:
    """
    Retrieve all existing website components

    Returns:
        Dictionary of components indexed by name
    """
    # First check our current context (faster than loading from DB)
    if ctx.deps.components:
        return {k: v.dict() for k, v in ctx.deps.components.items()}

    # Look for components in the database
    memory_ids = ctx.deps.db.list_all()
    components = {}

    for memory_id in memory_ids:
        node = ctx.deps.db.load(memory_id)
        if node.get_attribute("memory_type") == "component":
            name = node.get_attribute("name")
            components[name] = {
                "name": name,
                "type": node.get_attribute("type"),
                "code": node.get_attribute("code"),
                "description": node.get_attribute("description")
            }

            # Update our context too for future use
            ctx.deps.components[name] = WebsiteComponent(
                name=name,
                type=node.get_attribute("type"),
                code=node.get_attribute("code"),
                description=node.get_attribute("description")
            )

    return components


@website_generator.result_validator
async def ensure_correct_component_format(
    ctx: RunContext[AgentContext],
    result: AgentResponse
) -> bool:
    """Validate that generated components have proper code format."""
    for component in result.generated_components:
        if not component.code or len(component.code.strip()) < 10:
            raise ModelRetry(f"Component '{component.name}' has no code or very little code. Please provide complete code.")

        # Ensure Python code has proper imports
        if component.type in ["route", "model", "form", "main"] and "import" not in component.code.lower():
            raise ModelRetry(f"Component '{component.name}' is missing proper imports. Please add necessary imports.")

        # Ensure HTML templates have basic structure
        if component.type == "template" and (
            "<html" not in component.code.lower() or
            "</html>" not in component.code.lower()
        ):
            raise ModelRetry(f"HTML template '{component.name}' is missing proper HTML structure. Please include <html> tags.")

    return True


async def process_user_message(user_message: str, context: AgentContext = None) -> Dict[str, Any]:
    """Process a user message through the agent"""

    # Create a new database if we don't have one yet
    if context is None:
        # Create a file-based database
        db = engramdb.Database.file_based(ENGRAMDB_PATH)

        context = AgentContext(db=db)

    # Store the user message in EngramDB
    vector = context.text_to_vector(user_message)
    node = engramdb.MemoryNode(vector)
    node.set_attribute("memory_type", "chat_message")
    node.set_attribute("role", "user")
    node.set_attribute("content", user_message)
    node.set_attribute("timestamp", datetime.now().isoformat())
    node.set_attribute("chat_id", str(context.current_chat_id))
    context.db.save(node)

    # Run the agent with the user message
    result = await website_generator.run(user_message, deps=context)

    # Store the agent's response in EngramDB
    vector = context.text_to_vector(result.data.message)
    node = engramdb.MemoryNode(vector)
    node.set_attribute("memory_type", "chat_message")
    node.set_attribute("role", "assistant")
    node.set_attribute("content", result.data.message)
    node.set_attribute("timestamp", datetime.now().isoformat())
    node.set_attribute("chat_id", str(context.current_chat_id))
    context.db.save(node)

    # If requested, save files
    if result.data.should_save_files:
        # Save files using the direct approach rather than through the RunContext
        try:
            # Create output directory if it doesn't exist
            os.makedirs(WEBSITE_OUTPUT_PATH, exist_ok=True)

            # Create typical Flask directory structure
            os.makedirs(os.path.join(WEBSITE_OUTPUT_PATH, "static"), exist_ok=True)
            os.makedirs(os.path.join(WEBSITE_OUTPUT_PATH, "templates"), exist_ok=True)

            # Fetch all component memories
            memory_ids = context.db.list_all()
            component_nodes = []

            for memory_id in memory_ids:
                node = context.db.load(memory_id)
                if node.get_attribute("memory_type") == "component":
                    component_nodes.append(node)

            # Track files we create
            created_files = []

            # Process each component and create the appropriate file
            for node in component_nodes:
                component_name = node.get_attribute("name")
                component_type = node.get_attribute("type")
                code = node.get_attribute("code")
                description = node.get_attribute("description")

                # Determine the filename and path
                if component_type == "route" or component_type == "main":
                    filename = "app.py" if component_name == "main" else f"{component_name}.py"
                    filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)
                elif component_type == "template":
                    filename = f"{component_name}.html"
                    filepath = os.path.join(WEBSITE_OUTPUT_PATH, "templates", filename)
                elif component_type == "static":
                    filename = f"{component_name}.css" if "css" in component_name else f"{component_name}.js"
                    filepath = os.path.join(WEBSITE_OUTPUT_PATH, "static", filename)
                elif component_type == "model":
                    filename = "models.py" if component_name == "models" else f"{component_name}_model.py"
                    filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)
                elif component_type == "form":
                    filename = "forms.py" if component_name == "forms" else f"{component_name}_form.py"
                    filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)
                else:
                    filename = f"{component_name}.py"
                    filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)

                # Write the file
                with open(filepath, 'w') as f:
                    f.write(code)

                created_files.append(filepath)

            # Create a basic requirements.txt with Flask dependencies
            requirements_content = """flask>=2.0.0
flask-wtf>=1.0.0
flask-sqlalchemy>=3.0.0
"""
            with open(os.path.join(WEBSITE_OUTPUT_PATH, "requirements.txt"), 'w') as f:
                f.write(requirements_content)

            created_files.append(os.path.join(WEBSITE_OUTPUT_PATH, "requirements.txt"))

            print(f"Created {len(created_files)} files in {WEBSITE_OUTPUT_PATH}")
        except Exception as e:
            print(f"Error saving files: {e}")

    # Return the agent's response
    return {
        "message": result.data.message,
        "generated_components": [comp.dict() for comp in result.data.generated_components],
        "next_steps": result.data.next_steps,
        "should_save_files": result.data.should_save_files
    }


def interactive_cli_chat():
    """Run an interactive CLI chat session with the website generator agent"""
    print("\n🌐 Welcome to the EngramDB + PydanticAI Flask Website Generator! 🌐\n")
    print("This agent helps you build Flask websites step by step while maintaining")
    print("context using EngramDB for memory and PydanticAI for structured reasoning.\n")
    print("Type 'exit' to quit, 'save' to save all files to disk.\n")

    # Initialize database
    if os.path.exists(ENGRAMDB_PATH):
        print(f"Using existing database at {ENGRAMDB_PATH}")
    else:
        print(f"Creating new database at {ENGRAMDB_PATH}")
    db = engramdb.Database.file_based(ENGRAMDB_PATH)

    # Create context
    context = AgentContext(db=db)

    # Display generated files path
    print(f"Generated files will be saved to: {WEBSITE_OUTPUT_PATH}\n")

    # Main chat loop
    while True:
        user_input = input("\nYou: ")

        # Check for exit command
        if user_input.lower() in ['exit', 'quit']:
            print("\nThank you for using the Flask Website Generator. Goodbye!")
            break

        # Check for save command
        if user_input.lower() == 'save':
            print("\nSaving all generated files to disk...")
            # Instead of using RunContext directly, let's use a simpler approach
            try:
                # Create output directory if it doesn't exist
                os.makedirs(WEBSITE_OUTPUT_PATH, exist_ok=True)

                # Create typical Flask directory structure
                os.makedirs(os.path.join(WEBSITE_OUTPUT_PATH, "static"), exist_ok=True)
                os.makedirs(os.path.join(WEBSITE_OUTPUT_PATH, "templates"), exist_ok=True)

                # Fetch all component memories
                memory_ids = context.db.list_all()
                component_nodes = []

                for memory_id in memory_ids:
                    node = context.db.load(memory_id)
                    if node.get_attribute("memory_type") == "component":
                        component_nodes.append(node)

                # Track files we create
                created_files = []

                # Process each component and create the appropriate file
                for node in component_nodes:
                    component_name = node.get_attribute("name")
                    component_type = node.get_attribute("type")
                    code = node.get_attribute("code")
                    description = node.get_attribute("description")

                    # Determine the filename and path
                    if component_type == "route" or component_type == "main":
                        filename = "app.py" if component_name == "main" else f"{component_name}.py"
                        filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)
                    elif component_type == "template":
                        filename = f"{component_name}.html"
                        filepath = os.path.join(WEBSITE_OUTPUT_PATH, "templates", filename)
                    elif component_type == "static":
                        filename = f"{component_name}.css" if "css" in component_name else f"{component_name}.js"
                        filepath = os.path.join(WEBSITE_OUTPUT_PATH, "static", filename)
                    elif component_type == "model":
                        filename = "models.py" if component_name == "models" else f"{component_name}_model.py"
                        filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)
                    elif component_type == "form":
                        filename = "forms.py" if component_name == "forms" else f"{component_name}_form.py"
                        filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)
                    else:
                        filename = f"{component_name}.py"
                        filepath = os.path.join(WEBSITE_OUTPUT_PATH, filename)

                    # Write the file
                    with open(filepath, 'w') as f:
                        f.write(code)

                    created_files.append(filepath)

                # Create a basic requirements.txt with Flask dependencies
                requirements_content = """flask>=2.0.0
flask-wtf>=1.0.0
flask-sqlalchemy>=3.0.0
"""
                with open(os.path.join(WEBSITE_OUTPUT_PATH, "requirements.txt"), 'w') as f:
                    f.write(requirements_content)

                created_files.append(os.path.join(WEBSITE_OUTPUT_PATH, "requirements.txt"))

                print(f"Created {len(created_files)} files in {WEBSITE_OUTPUT_PATH}:")
                for file in created_files:
                    print(f"  - {file}")
            except Exception as e:
                print(f"Error saving files: {e}")
            continue

        # Process the message
        print("\nThinking...")
        response = asyncio.run(process_user_message(user_input, context))

        # Display the response
        print("\nAgent:", response["message"])

        # Display generated components if any
        if response["generated_components"]:
            print("\nGenerated Components:")
            for comp in response["generated_components"]:
                print(f"  - {comp['name']} ({comp['type']}): {comp['description']}")

        # Display next steps if any
        if response["next_steps"]:
            print("\nSuggested next steps:")
            for step in response["next_steps"]:
                print(f"  - {step}")

        # If files were saved
        if response["should_save_files"]:
            print(f"\nFiles saved to: {WEBSITE_OUTPUT_PATH}")


if __name__ == "__main__":
    interactive_cli_chat()