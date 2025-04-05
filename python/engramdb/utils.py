"""
Utility functions for the EngramDB Python package.
"""

import time
import numpy as np
from typing import List, Optional, Union, Dict, Any

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        a: First vector
        b: Second vector
        
    Returns:
        Cosine similarity, a value between -1 and 1
    """
    if len(a) != len(b):
        raise ValueError("Vectors must have the same length")
    
    a_array = np.array(a)
    b_array = np.array(b)
    
    norm_a = np.linalg.norm(a_array)
    norm_b = np.linalg.norm(b_array)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return np.dot(a_array, b_array) / (norm_a * norm_b)

def format_timestamp(timestamp: int, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Format a Unix timestamp as a human-readable string.
    
    Args:
        timestamp: Unix timestamp in seconds
        format_str: Format string (default: "%Y-%m-%d %H:%M:%S")
        
    Returns:
        Formatted timestamp string
    """
    return time.strftime(format_str, time.localtime(timestamp))

def format_duration(seconds: int) -> str:
    """
    Format a duration in seconds as a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string
    """
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    
    minutes = seconds // 60
    if minutes < 60:
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    
    hours = minutes // 60
    minutes = minutes % 60
    if hours < 24:
        if minutes == 0:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        return f"{hours} hour{'s' if hours != 1 else ''} {minutes} minute{'s' if minutes != 1 else ''}"
    
    days = hours // 24
    hours = hours % 24
    if hours == 0:
        return f"{days} day{'s' if days != 1 else ''}"
    return f"{days} day{'s' if days != 1 else ''} {hours} hour{'s' if hours != 1 else ''}"

def now() -> int:
    """
    Get the current Unix timestamp in seconds.
    
    Returns:
        Current Unix timestamp
    """
    return int(time.time())

def timestamp_ago(seconds: int = 0, minutes: int = 0, hours: int = 0, days: int = 0) -> int:
    """
    Get a Unix timestamp for a time in the past.
    
    Args:
        seconds: Seconds ago (default: 0)
        minutes: Minutes ago (default: 0)
        hours: Hours ago (default: 0)
        days: Days ago (default: 0)
        
    Returns:
        Unix timestamp for the specified time ago
    """
    total_seconds = seconds + minutes * 60 + hours * 3600 + days * 86400
    return now() - total_seconds