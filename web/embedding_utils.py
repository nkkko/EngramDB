"""
Embedding utilities for the EngramDB web interface.
Provides functionality to generate vector embeddings from text using the 
multilingual-e5-large-instruct model.
"""
import os
import logging
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable to hold the model
_model = None
_tokenizer = None

def get_embedding_model():
    """
    Loads the multilingual-e5-large-instruct model.
    Returns the model and tokenizer.
    """
    global _model, _tokenizer
    
    # Return cached model if already loaded
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    
    try:
        from transformers import AutoModel, AutoTokenizer
        import torch
        
        logger.info("Loading multilingual-e5-large-instruct model...")
        model_name = "intfloat/multilingual-e5-large-instruct"
        
        # Add cache directory to prevent permission issues
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        _tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        _model = AutoModel.from_pretrained(model_name, cache_dir=cache_dir)

        # Check if CUDA is available and move model to GPU if possible
        if torch.cuda.is_available():
            logger.info("CUDA is available. Moving model to GPU.")
            _model = _model.to("cuda")
        
        logger.info("Model loaded successfully.")
        return _model, _tokenizer
    
    except Exception as e:
        logger.error(f"Error loading embedding model: {e}")
        return None, None

def generate_embeddings(text, instruction=None):
    """
    Generate embeddings from text using the multilingual-e5-large-instruct model.
    
    Args:
        text (str): The text to encode
        instruction (str, optional): The instruction to prepend (e.g., "Find similar text").
                                    If None, uses default "represent the document for retrieval"
    
    Returns:
        numpy.ndarray: The embedding vector
        None: If there was an error generating embeddings
    """
    try:
        import torch
        
        model, tokenizer = get_embedding_model()
        if model is None or tokenizer is None:
            logger.warning("Model not available, returning None")
            return None
        
        # Format input according to model expectations
        if instruction:
            input_text = f"instruction: {instruction} text: {text}"
        else:
            input_text = f"query: {text}"  # Default format for retrieval
            
        # Tokenize and prepare for model
        inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        
        # Move inputs to the same device as the model
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # Generate embeddings
        with torch.no_grad():
            outputs = model(**inputs)
            # Use CLS token embedding (first token)
            embeddings = outputs.last_hidden_state[:, 0, :]
            
        # Normalize embeddings
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        # Convert to numpy and return the first embedding (batch size 1)
        return embeddings[0].cpu().numpy()
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return None

def generate_embedding_from_query(query_text):
    """
    Generate an embedding for a search query.
    
    Args:
        query_text (str): The search query text
        
    Returns:
        numpy.ndarray: The embedding vector
    """
    return generate_embeddings(query_text, instruction="Find similar text")

def generate_embedding_for_memory(memory_text, category=None):
    """
    Generate an embedding for storing in memory.
    
    Args:
        memory_text (str): The memory text content
        category (str, optional): The category of the memory
        
    Returns:
        numpy.ndarray: The embedding vector
    """
    instruction = f"Represent the {category} document for retrieval" if category else "Represent the document for retrieval"
    return generate_embeddings(memory_text, instruction=instruction)

def mock_embeddings(dimensions=384):
    """
    Generate random mock embeddings when the model is not available.
    
    Args:
        dimensions (int): The dimensions of the mock embedding vector
        
    Returns:
        numpy.ndarray: A random embedding vector with normalized values
    """
    # Generate random embeddings in the same shape as the real model output
    embedding = np.random.normal(0, 1, dimensions)
    # Normalize
    embedding = embedding / np.linalg.norm(embedding)
    return embedding