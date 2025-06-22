"""
Stylistic embedding module
Creates embeddings for writing style and linguistic patterns
"""

import numpy as np
from typing import List, Dict, Any, Optional
import re
from collections import Counter


class StyleEmbedder:
    """Creates stylistic embeddings for text analysis"""
    
    def __init__(self):
        """Initialize style embedder"""
        self.feature_names = [
            'avg_sentence_length', 'avg_word_length', 'punctuation_ratio',
            'question_ratio', 'exclamation_ratio', 'caps_ratio',
            'first_person_ratio', 'emotion_words_ratio', 'complexity_score'
        ]
    
    def extract_style_features(self, text: str) -> Dict[str, float]:
        """
        Extract stylistic features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of style features
        """
        if not text:
            return {name: 0.0 for name in self.feature_names}
        
        # Basic text statistics
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        words = text.split()
        
        features = {}
        
        # Average sentence length
        features['avg_sentence_length'] = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        
        # Average word length
        features['avg_word_length'] = np.mean([len(word) for word in words]) if words else 0
        
        # Punctuation ratio
        punct_count = len(re.findall(r'[.,!?;:]', text))
        features['punctuation_ratio'] = punct_count / len(text) if text else 0
        
        # Question and exclamation ratios
        features['question_ratio'] = text.count('?') / len(sentences) if sentences else 0
        features['exclamation_ratio'] = text.count('!') / len(sentences) if sentences else 0
        
        # Capitalization ratio
        caps_count = sum(1 for c in text if c.isupper())
        features['caps_ratio'] = caps_count / len(text) if text else 0
        
        # First person pronouns
        first_person = ['i', 'me', 'my', 'mine', 'myself']
        first_person_count = sum(1 for word in words if word.lower() in first_person)
        features['first_person_ratio'] = first_person_count / len(words) if words else 0
        
        # Emotion words (simple list)
        emotion_words = ['love', 'hate', 'happy', 'sad', 'angry', 'excited', 'worried', 'afraid']
        emotion_count = sum(1 for word in words if word.lower() in emotion_words)
        features['emotion_words_ratio'] = emotion_count / len(words) if words else 0
        
        # Complexity score (based on sentence and word length variation)
        if sentences and words:
            sent_lengths = [len(s.split()) for s in sentences]
            word_lengths = [len(word) for word in words]
            complexity = np.std(sent_lengths) + np.std(word_lengths)
            features['complexity_score'] = complexity / 10  # Normalize
        else:
            features['complexity_score'] = 0
        
        return features
    
    def create_embedding(self, text: str) -> np.ndarray:
        """
        Create stylistic embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Numpy array containing the style embedding
        """
        features = self.extract_style_features(text)
        embedding = np.array([features[name] for name in self.feature_names])
        
        # Normalize the embedding
        if np.linalg.norm(embedding) > 0:
            embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def create_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """
        Create style embeddings for multiple texts
        
        Args:
            texts: List of input texts
            
        Returns:
            List of style embedding arrays
        """
        return [self.create_embedding(text) for text in texts]


def create_style_embedding(text: str, embedder: Optional[StyleEmbedder] = None) -> np.ndarray:
    """
    Create stylistic embedding for text
    
    Args:
        text: Input text
        embedder: Optional embedder instance
        
    Returns:
        Style embedding array
    """
    if embedder is None:
        embedder = StyleEmbedder()
    
    return embedder.create_embedding(text)


def compute_style_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute similarity between two style embeddings
    
    Args:
        emb1: First style embedding
        emb2: Second style embedding
        
    Returns:
        Style similarity score
    """
    if emb1.size == 0 or emb2.size == 0:
        return 0.0
    
    # Use cosine similarity for style embeddings too
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
    
    similarity = np.dot(emb1_norm, emb2_norm)
    return float(similarity)