"""
Semantic embedding module
Creates embeddings for semantic content analysis
"""

import numpy as np
from typing import List, Dict, Any, Optional
import hashlib


class SemanticEmbedder:
    """Creates semantic embeddings for text content"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic embedder
        
        Args:
            model_name: Name of the embedding model to use
        """
        self.model_name = model_name
        self._model = None
        
    def _load_model(self):
        """Lazy load the embedding model"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
            except ImportError:
                raise ImportError("sentence-transformers package is required for semantic embeddings")
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create semantic embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array containing the embedding
        """
        if not text:
            return np.zeros(384)  # Default embedding size for MiniLM
        
        self._load_model()
        embedding = self._model.encode([text])[0]
        return embedding
    
    def create_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of embedding arrays
        """
        if not texts:
            return []
        
        self._load_model()
        embeddings = self._model.encode(texts)
        return [emb for emb in embeddings]


def create_semantic_embedding(text: str, embedder: Optional[SemanticEmbedder] = None) -> np.ndarray:
    """
    Create semantic embedding for text
    
    Args:
        text: Input text
        embedder: Optional embedder instance
        
    Returns:
        Semantic embedding array
    """
    if embedder is None:
        embedder = SemanticEmbedder()
    
    return embedder.create_embedding(text)


def compute_semantic_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        emb1: First embedding
        emb2: Second embedding
        
    Returns:
        Cosine similarity score
    """
    if emb1.size == 0 or emb2.size == 0:
        return 0.0
    
    # Normalize embeddings
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
    
    # Compute cosine similarity
    similarity = np.dot(emb1_norm, emb2_norm)
    return float(similarity)