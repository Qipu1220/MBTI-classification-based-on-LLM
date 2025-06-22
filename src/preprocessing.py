"""
Text preprocessing module
Handles Unicode NFKC normalization, light cleanup, and text chunking
"""

import unicodedata
import re
from typing import List, Dict, Any


def preprocess_text(text: str, use_unidecode: bool = False) -> str:
    """
    Preprocess text with Unicode NFKC normalization and light cleanup
    
    Args:
        text: Raw input text
        use_unidecode: Whether to apply unidecode for ASCII normalization
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # 1. Unicode NFKC normalization
    text = unicodedata.normalize('NFKC', text)
    
    # 2. Light Token Clean-up
    # Convert to lowercase
    text = text.lower()
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # If unidecode is requested, apply it
    if use_unidecode:
        try:
            from unidecode import unidecode
            text = unidecode(text)
        except ImportError:
            print("Warning: unidecode package not found. Skipping ASCII normalization.")
    
    # Keep emojis and basic punctuation
    # Only normalize excessive punctuation
    text = re.sub(r'([.,!?;:]){2,}', r'\1', text)
    
    return text


def chunk_text(text: str, num_chunks: int = 5, min_chunk_size: int = 100) -> List[str]:
    """
    Split text into exactly 5 chunks with balanced sizes
    
    Args:
        text: Input text to chunk
        num_chunks: Number of chunks to create (default 5)
        min_chunk_size: Minimum size for each chunk
        
    Returns:
        List of 5 text chunks
    """
    if not text:
        return []
    
    # Split text into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    
    # Calculate target chunk size
    total_tokens = sum(len(s.split()) for s in sentences)
    target_size = max(min_chunk_size, total_tokens // num_chunks)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence_tokens = len(sentence.split())
        
        if current_size + sentence_tokens <= target_size:
            current_chunk.append(sentence)
            current_size += sentence_tokens
        else:
            if current_chunk:
                chunks.append(' '.join(current_chunk).strip())
            current_chunk = [sentence]
            current_size = sentence_tokens
    
    # Add last chunk
    if current_chunk:
        chunks.append(' '.join(current_chunk).strip())
    
    # If we have more than 5 chunks, merge smaller ones
    while len(chunks) > num_chunks:
        # Find smallest chunk to merge
        min_idx = min(range(len(chunks)), key=lambda i: len(chunks[i].split()))
        
        # Merge with previous chunk if possible
        if min_idx > 0:
            chunks[min_idx-1] += ' ' + chunks.pop(min_idx)
        else:
            chunks[0] += ' ' + chunks.pop(1)
    
    # If we have less than 5 chunks, split larger ones
    while len(chunks) < num_chunks and chunks:
        # Find largest chunk to split
        max_idx = max(range(len(chunks)), key=lambda i: len(chunks[i].split()))
        
        # Split it into two parts
        sentences = chunks[max_idx].split('. ')
        mid = len(sentences) // 2
        
        if mid > 0:
            chunks[max_idx] = '. '.join(sentences[:mid]) + '.'
            chunks.insert(max_idx + 1, '. '.join(sentences[mid:]))
    
    # Ensure minimum chunk size
    for i, chunk in enumerate(chunks):
        if len(chunk.split()) < min_chunk_size:
            if i > 0:
                chunks[i-1] += ' ' + chunk
                del chunks[i]
            elif len(chunks) > 1:
                chunks[1] = chunk + ' ' + chunks[1]
                del chunks[0]
    
    # If we still have less than 5 chunks, duplicate the last one
    while len(chunks) < num_chunks:
        chunks.append(chunks[-1])
    
    # Take only the first 5 chunks
    return chunks[:num_chunks]


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