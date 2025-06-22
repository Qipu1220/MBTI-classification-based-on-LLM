"""
Text preprocessing module
Handles Unicode NFKC normalization, light cleanup, and text chunking
"""

import unicodedata
import re
from typing import List, Dict, Any


def preprocess_text(text: str) -> str:
    """
    Preprocess text with Unicode NFKC normalization and light cleanup
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    
    # Light cleanup
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?;:\-()"\']', ' ', text)
    
    # Remove excessive punctuation
    text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
    
    return text


def chunk_text(text: str, max_chunk_size: int = 512, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better processing
    
    Args:
        text: Input text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    if not text or len(text) <= max_chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    
    while start < len(text):
        # Calculate end position
        end = start + max_chunk_size
        
        # If this is not the last chunk, try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings within the last 100 characters
            sentence_end = -1
            for i in range(max(0, end - 100), end):
                if text[i] in '.!?':
                    sentence_end = i + 1
            
            if sentence_end > start:
                end = sentence_end
        
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        
        # Move start position with overlap
        start = max(start + 1, end - overlap)
        
        # Prevent infinite loop
        if start >= len(text):
            break
    
    return chunks


def clean_mbti_response(response: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean MBTI response data
    
    Args:
        response: Raw MBTI response dictionary
        
    Returns:
        Cleaned response dictionary
    """
    cleaned = {}
    
    for key, value in response.items():
        if isinstance(value, str):
            cleaned[key] = preprocess_text(value)
        else:
            cleaned[key] = value
    
    return cleaned