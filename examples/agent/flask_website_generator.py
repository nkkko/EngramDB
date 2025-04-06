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
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext, ModelRetry
import engramdb_py as engramdb

# Load environment variables from .env file if it exists
env_path = Path('.env')
if env_path.exists():
    print(f"Loading configuration from {env_path}")
    with open(env_path) as f:
        for line in f:
            # Skip comments and empty lines
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            # Parse key=value pairs
            if '=' in line:
                key, value = line.split('=', 1)
                # Remove quotes if present
                value = value.strip('"\'')
                if value and key not in os.environ:
                    os.environ[key] = value
                    print(f"  Set {key}")

# Configuration
ENGRAMDB_PATH = os.environ.get("ENGRAMDB_PATH", "agent_memory.engramdb")
WEBSITE_OUTPUT_PATH = os.environ.get("WEBSITE_OUTPUT_PATH", os.path.join(tempfile.gettempdir(), "generated_flask_website"))
# Try different models in order of availability, use environment variable if set
DEFAULT_MODEL = os.environ.get("PYDANTIC_AI_MODEL")

# If not explicitly set, try to pick a suitable default based on available keys
if not DEFAULT_MODEL:
    if "OPENAI_API_KEY" in os.environ:
        DEFAULT_MODEL = "openai:gpt-3.5-turbo"  # Less expensive model for testing
    elif "ANTHROPIC_API_KEY" in os.environ:
        DEFAULT_MODEL = "anthropic:claude-3-5-haiku-latest"  # More widely available model
    elif "GROQ_API_KEY" in os.environ:
        DEFAULT_MODEL = "groq:llama-3.1-8b-instant"
    else:
        DEFAULT_MODEL = "openai:gpt-3.5-turbo"  # Default if no explicit choice


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


# No need to adjust model name since we're using the latest naming conventions

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
    
    IMPORTANT: You MUST respond with a valid JSON object that matches the AgentResponse type with these fields:
    - message: string (your response to the user)
    - generated_components: array of WebsiteComponent objects (if any)
    - should_save_files: boolean
    - next_steps: array of strings

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

    # Connect to relevant requirements - with proper error handling
    try:
        for req in ctx.deps.user_requirements:
            try:
                req_id = uuid.UUID(req["id"])
                # Convert IDs to strings if needed - some versions of EngramDB may require string IDs
                ctx.deps.db.connect(str(req_id), str(component_id), "fulfilled_by", 0.8)
            except Exception as e:
                print(f"Warning: Could not connect requirement to component: {e}")
    except Exception as e:
        print(f"Warning: Error connecting requirements: {e}")

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


def parse_anthropic_response(result) -> AgentResponse:
    """Helper function to parse Anthropic model responses that might not be properly structured.
    This helps handle cases where the model returns a boolean or other unexpected type."""
    
    if isinstance(result.data, bool) or result.data is None:
        print(f"Attempting to reconstruct response from the raw completion")
        # Try to extract structured data from the raw completion text
        try:
            # Get the raw completion text
            raw_text = result.completion if hasattr(result, 'completion') else ""
            
            # Look for JSON in the response
            import json
            import re
            
            # Try to find JSON pattern in the text
            json_match = re.search(r'```json\n(.*?)\n```', raw_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                parsed_data = json.loads(json_str)
                return AgentResponse(**parsed_data)
            
            # If no JSON block found, try to parse the entire text if it looks like JSON
            if raw_text.strip().startswith('{') and raw_text.strip().endswith('}'):
                try:
                    parsed_data = json.loads(raw_text)
                    return AgentResponse(**parsed_data)
                except:
                    pass
            
            # Extract a message at minimum
            message = raw_text
            if len(message) > 500:  # Truncate if very long
                message = message[:500] + "..."
                
            return AgentResponse(
                message=message,
                generated_components=[],
                next_steps=["Please provide more details about your website requirements"],
                should_save_files=False
            )
        except Exception as e:
            print(f"Error parsing raw completion: {e}")
            # Return a default response
            return AgentResponse(
                message="I'm having trouble understanding. Could you provide more details about the website you'd like to build?",
                generated_components=[],
                next_steps=["Describe the purpose of your website", "Specify any specific features you need"],
                should_save_files=False
            )
    else:
        # Return the correctly structured response
        return result.data


async def process_user_message(user_message: str, context: AgentContext = None) -> Dict[str, Any]:
    """Process a user message through the agent"""

    # Create a new database if we don't have one yet
    if context is None:
        try:
            # Create a file-based database
            db = engramdb.Database.file_based(ENGRAMDB_PATH)
            context = AgentContext(db=db)
        except Exception as e:
            print(f"Error creating database: {e}")
            # Create a fallback in-memory database
            db = engramdb.Database.in_memory()
            context = AgentContext(db=db)

    try:
        # Store the user message in EngramDB - with error handling
        try:
            vector = context.text_to_vector(user_message)
            node = engramdb.MemoryNode(vector)
            node.set_attribute("memory_type", "chat_message")
            node.set_attribute("role", "user")
            node.set_attribute("content", user_message)
            node.set_attribute("timestamp", datetime.now().isoformat())
            node.set_attribute("chat_id", str(context.current_chat_id))
            context.db.save(node)
        except Exception as e:
            print(f"Warning: Could not store user message: {e}")

        # Run the agent with the user message - with better error handling for model issues
        try:
            # Print which model we're using
            print(f"Using model: {DEFAULT_MODEL}")

            # Import the needed modules for custom initialization
            from pydantic_ai.providers.openai import OpenAIProvider
            from pydantic_ai.models.openai import OpenAIModel
            
            # Check if we're using OpenAI and have a project key
            custom_model = None
            if "openai:" in DEFAULT_MODEL and "OPENAI_API_KEY" in os.environ:
                api_key = os.environ["OPENAI_API_KEY"]
                if api_key.startswith("sk-proj-"):
                    print("Using a custom provider for project API key")
                    model_name = DEFAULT_MODEL.split(":", 1)[1]
                    # Create a custom provider that can handle project keys
                    provider = OpenAIProvider(api_key=api_key)
                    custom_model = OpenAIModel(model_name, provider=provider)
            
            # Use either the custom model or the default configuration
            if custom_model:
                result = await website_generator.run(user_message, model=custom_model, deps=context)
            else:
                result = await website_generator.run(user_message, deps=context)
            
            # Debug the result structure
            print(f"DEBUG: Result type: {type(result)}")
            print(f"DEBUG: Result.data type: {type(result.data)}")
            
            # Try to parse the response if it's not properly structured
            if "anthropic:" in DEFAULT_MODEL and (isinstance(result.data, bool) or not hasattr(result.data, 'message')):
                print("Attempting to fix Anthropic response format...")
                result.data = parse_anthropic_response(result)
                print(f"After fix - Result.data type: {type(result.data)}")
        except Exception as e:
            print(f"Error with model {DEFAULT_MODEL}: {e}")
            
            # Try a fallback to Anthropic with a different model name
            if "anthropic:" in DEFAULT_MODEL and "ANTHROPIC_API_KEY" in os.environ:
                try:
                    print("Trying fallback to Anthropic Claude 3 Haiku (latest)...")
                    # Temporarily modify the agent's model
                    orig_model = website_generator.model
                    from pydantic_ai.models.anthropic import AnthropicModel
                    from pydantic_ai.providers.anthropic import AnthropicProvider
                    
                    # Use the latest model
                    api_key = os.environ["ANTHROPIC_API_KEY"]
                    provider = AnthropicProvider(api_key=api_key)
                    fallback_model = AnthropicModel("claude-3-5-haiku-latest", provider=provider)
                    website_generator.model = fallback_model

                    # Try again with Anthropic but with correct model name
                    result = await website_generator.run(user_message, model=fallback_model, deps=context)

                    # Debug the result structure
                    print(f"DEBUG: Result type with fallback: {type(result)}")
                    print(f"DEBUG: Result.data type with fallback: {type(result.data)}")
                    
                    # Try to parse the response with our helper function
                    if isinstance(result.data, bool) or not hasattr(result.data, 'message'):
                        print("Attempting to fix fallback Anthropic response format...")
                        result.data = parse_anthropic_response(result)
                        print(f"After fallback fix - Result.data type: {type(result.data)}")

                    # Restore the original model
                    website_generator.model = orig_model
                except Exception as fallback_error:
                    print(f"Fallback also failed: {fallback_error}")
                    # Raise a more helpful error
                    raise Exception(f"Failed with main model ({DEFAULT_MODEL}) and fallback. Please check your API keys and connectivity.") from e
            else:
                # Re-raise the original error
                raise

        # Check if result is properly formatted
        if isinstance(result.data, bool) or not hasattr(result.data, 'message'):
            print(f"Warning: Agent result has unexpected type: {type(result.data)}. Using default message.")
            print(f"Raw result: {result.data}")
            response_message = "I'm having trouble understanding. Could you please provide more details about the website you'd like to build?"
            generated_components = []
            next_steps = ["Describe the purpose of your website", "Specify any specific features you need"]
            should_save_files = False
        else:
            # Use the structured response
            response_message = result.data.message
            generated_components = result.data.generated_components if hasattr(result.data, 'generated_components') else []
            next_steps = result.data.next_steps if hasattr(result.data, 'next_steps') else []
            should_save_files = result.data.should_save_files if hasattr(result.data, 'should_save_files') else False

        # Store the agent's response in EngramDB
        vector = context.text_to_vector(response_message)
        node = engramdb.MemoryNode(vector)
        node.set_attribute("memory_type", "chat_message")
        node.set_attribute("role", "assistant")
        node.set_attribute("content", response_message)
        node.set_attribute("timestamp", datetime.now().isoformat())
        node.set_attribute("chat_id", str(context.current_chat_id))
        context.db.save(node)

        # If requested, save files
        if should_save_files:
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
            "message": response_message,
            "generated_components": [comp.dict() if hasattr(comp, 'dict') else comp for comp in generated_components],
            "next_steps": next_steps,
            "should_save_files": should_save_files
        }
    except Exception as e:
        print(f"Error processing message: {e}")
        # Return a fallback response
        return {
            "message": "I encountered an error while processing your request. Please try again.",
            "generated_components": [],
            "next_steps": ["Describe your website requirements again"],
            "should_save_files": False
        }


def verify_model_access():
    """Verify that we can access the specified model"""
    try:
        if "openai:gpt" in DEFAULT_MODEL:
            if "OPENAI_API_KEY" not in os.environ:
                print("⚠️  Warning: Using OpenAI model but OPENAI_API_KEY environment variable is not set")
                return False
            else:
                # Basic format check for OpenAI keys
                key = os.environ["OPENAI_API_KEY"]
                # OpenAI now uses project-specific keys that start with sk-proj-
                if not (key.startswith("sk-") or key.startswith("sk-proj-")):
                    print("⚠️  Warning: Your OpenAI API key doesn't follow the expected format")
                    print("   Valid formats: sk-... (older) or sk-proj-... (newer project keys)")
                # Don't print the actual key for security reasons
                print(f"   API Key: {key[:5]}...{key[-4:]} (length: {len(key)})")
                
        elif "anthropic:claude" in DEFAULT_MODEL:
            if "ANTHROPIC_API_KEY" not in os.environ:
                print("⚠️  Warning: Using Anthropic model but ANTHROPIC_API_KEY environment variable is not set")
                return False
            else:
                # Check Anthropic key format
                key = os.environ["ANTHROPIC_API_KEY"]
                if not key.startswith("sk-ant-"):
                    print("⚠️  Warning: Your Anthropic API key doesn't follow the expected format (should start with sk-ant-)")
                print(f"   API Key: {key[:7]}...{key[-4:]} (length: {len(key)})")
                
        elif "groq:" in DEFAULT_MODEL:
            if "GROQ_API_KEY" not in os.environ:
                print("⚠️  Warning: Using Groq model but GROQ_API_KEY environment variable is not set")
                return False
            else:
                key = os.environ["GROQ_API_KEY"]
                print(f"   API Key: {key[:5]}...{key[-4:]} (length: {len(key)})")
                
        print(f"✅ Model configuration: {DEFAULT_MODEL}")
        
        # Add explicit instructions for fixing API key issues
        print("\nTo fix API key issues:")
        print("1. Make sure your API key is correct and active")
        print("2. Set it correctly in your .env file or environment variables")
        print("3. For OpenAI: https://platform.openai.com/account/api-keys")
        print("4. For Anthropic: https://console.anthropic.com/settings/keys")
        print("5. For Groq: https://console.groq.com/keys\n")
        
        return True
    except Exception as e:
        print(f"⚠️  Warning: Could not verify model access: {e}")
        return False


def interactive_cli_chat():
    """Run an interactive CLI chat session with the website generator agent"""
    print("\n🌐 Welcome to the EngramDB + PydanticAI Flask Website Generator! 🌐\n")
    print("This agent helps you build Flask websites step by step while maintaining")
    print("context using EngramDB for memory and PydanticAI for structured reasoning.\n")
    print("Type 'exit' to quit, 'save' to save all files to disk.\n")

    # Verify model access
    verify_model_access()

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