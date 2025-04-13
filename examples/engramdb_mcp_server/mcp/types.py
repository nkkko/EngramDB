"""
Mock MCP Types module
"""

# Define schema types
JSONSchema = dict
JSONType = dict

# Define basic types
class ToolDefinition:
    def __init__(self, name: str, description: str, input_schema: JSONSchema):
        self.name = name
        self.description = description
        self.input_schema = input_schema