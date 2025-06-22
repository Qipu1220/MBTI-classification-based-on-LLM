"""
Semantic embedding module
Creates embeddings for semantic content analysis
"""

import numpy as np
import torch
import logging
from typing import List, Dict, Any, Optional, Union
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('semantic_embedding.log')
    ]
)
logger = logging.getLogger(__name__)


class SemanticEmbedder:
    """Creates semantic embeddings for text content"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize semantic embedder
        
        Args:
            model_name: Name of the embedding model to use (default: "all-MiniLM-L6-v2")
            device: Device to run the model on ('cuda' or 'cpu'). If None, will auto-detect.
        """
        self.model_name = model_name
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self._model = None
        logger.info(f"Initialized SemanticEmbedder with model '{model_name}' on device '{self.device}'")
        
    def _load_model(self):
        """Lazy load the embedding model with error handling"""
        if self._model is not None:
            return
            
        try:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading model '{self.model_name}' on device '{self.device}'...")
            
            # Set device for sentence-transformers
            if self.device.startswith('cuda') and not torch.cuda.is_available():
                logger.warning("CUDA not available, falling back to CPU")
                self.device = 'cpu'
                
            self._model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Successfully loaded model '{self.model_name}' on {self.device}")
            
        except ImportError:
            logger.warning("sentence-transformers không khả dụng. Sẽ dùng vector 0 thay thế.")
            self._model = None
        except Exception as e:
            logger.warning(f"Không thể tải mô hình '{self.model_name}': {str(e)}. Sẽ dùng vector 0 thay thế.")
            self._model = None
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create semantic embedding for text
        
        Args:
            text: Input text (can be empty or None)
            
        Returns:
            Numpy array containing the embedding (zeros if text is empty or error occurs)
        """
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Empty or invalid text provided, returning zero vector")
            return np.zeros(384)  # Default embedding size for MiniLM
            
        try:
            self._load_model()
            text = text.strip()
            
            # Encode the text
            if self._model is None:
                return np.zeros(384)
            with torch.no_grad():
                embedding = self._model.encode([text], convert_to_numpy=True, show_progress_bar=False)[0]
                
            if not isinstance(embedding, np.ndarray):
                embedding = embedding.cpu().numpy()
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error creating embedding: {str(e)}")
            return np.zeros(384)  # Return zero vector on error
    
    def create_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create embeddings for multiple texts with error handling
        
        Args:
            texts: List of input texts (can contain None or empty strings)
            
        Returns:
            List of numpy arrays containing the embeddings (zeros for invalid inputs)
        """
        if not texts or not isinstance(texts, list):
            logger.warning("Invalid input texts, returning empty list")
            return []
            
        # Filter out invalid texts
        valid_texts = []
        text_indices = []
        for i, text in enumerate(texts):
            if text and isinstance(text, str) and text.strip():
                valid_texts.append(text.strip())
                text_indices.append(i)
        
        if not valid_texts:
            return [np.zeros(384) for _ in texts]
            
        try:
            self._load_model()
            
            # Process in batches to avoid OOM
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch = valid_texts[i:i+batch_size]
                with torch.no_grad():
                    batch_embeddings = self._model.encode(
                        batch, 
                        convert_to_numpy=True, 
                        show_progress_bar=False,
                        batch_size=len(batch)
                    )
                all_embeddings.append(batch_embeddings)
                
            # Concatenate all batch results
            if all_embeddings:
                valid_embeddings = np.concatenate(all_embeddings, axis=0)
            else:
                valid_embeddings = np.array([])
                
            # Create full results with zeros for invalid inputs
            results = [np.zeros(384) for _ in range(len(texts))]
            for idx, emb_idx in enumerate(text_indices):
                if idx < len(valid_embeddings):
                    results[emb_idx] = valid_embeddings[idx]
                    
            return results
            
        except Exception as e:
            logger.error(f"Error in create_embeddings_batch: {str(e)}")
            return [np.zeros(384) for _ in texts]


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