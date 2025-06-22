"""
Stylistic embedding module
Creates embeddings for writing style and linguistic patterns
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union
import re
import logging
from collections import Counter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('style_embedding.log')
    ]
)
logger = logging.getLogger(__name__)

class StyleEmbedder:
    """Creates stylistic embeddings for text analysis"""
    
    def __init__(self):
        """Initialize style embedder"""
        self.feature_names = [
            'avg_sentence_length', 'avg_word_length', 'punctuation_ratio',
            'question_ratio', 'exclamation_ratio', 'caps_ratio',
            'first_person_ratio', 'emotion_words_ratio', 'complexity_score'
        ]
    
    def extract_style_features(self, text: Union[str, None]) -> Dict[str, float]:
        """
        Extract stylistic features from text
        
        Args:
            text: Input text (can be None or empty)
            
        Returns:
            Dictionary of style features with default values if extraction fails
        """
        # Initialize default features
        default_features = {name: 0.0 for name in self.feature_names}
        
        # Input validation
        if not text or not isinstance(text, str) or not text.strip():
            logger.warning("Invalid or empty text provided to extract_style_features")
            return default_features
            
        try:
            text = text.strip()
            # Basic text statistics
            sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
            words = text.split()
            
            features = {}
            
            # Calculate each feature with individual error handling
            try:
                features['avg_sentence_length'] = float(np.mean([len(s.split()) for s in sentences])) if sentences else 0.0
            except Exception as e:
                logger.error(f"Error calculating avg_sentence_length: {str(e)}")
                features['avg_sentence_length'] = 0.0
                
            try:
                features['avg_word_length'] = float(np.mean([len(word) for word in words])) if words else 0.0
            except Exception as e:
                logger.error(f"Error calculating avg_word_length: {str(e)}")
                features['avg_word_length'] = 0.0
                
            try:
                punct_count = len(re.findall(r'[.,!?;:]', text))
                features['punctuation_ratio'] = float(punct_count) / len(text) if text else 0.0
            except Exception as e:
                logger.error(f"Error calculating punctuation_ratio: {str(e)}")
                features['punctuation_ratio'] = 0.0
                
            try:
                text_len = len(text)
                features['question_ratio'] = text.count('?') / text_len if text_len > 0 else 0.0
                features['exclamation_ratio'] = text.count('!') / text_len if text_len > 0 else 0.0
            except Exception as e:
                logger.error(f"Error calculating question/exclamation ratios: {str(e)}")
                features['question_ratio'] = 0.0
                features['exclamation_ratio'] = 0.0
                
            try:
                features['caps_ratio'] = sum(1 for c in text if c.isupper()) / len(text) if text else 0.0
            except Exception as e:
                logger.error(f"Error calculating caps_ratio: {str(e)}")
                features['caps_ratio'] = 0.0
                
            try:
                first_person = len(re.findall(r'\b(I|me|my|mine|myself)\b', text, re.IGNORECASE))
                features['first_person_ratio'] = first_person / len(words) if words else 0.0
            except Exception as e:
                logger.error(f"Error calculating first_person_ratio: {str(e)}")
                features['first_person_ratio'] = 0.0
                
            try:
                emotion_words = {'happy', 'sad', 'angry', 'excited', 'fear', 'love', 'hate', 
                              'joy', 'sadness', 'disgust', 'anger', 'surprise', 'trust'}
                emotion_count = sum(1 for word in words if word.lower() in emotion_words)
                features['emotion_words_ratio'] = emotion_count / len(words) if words else 0.0
            except Exception as e:
                logger.error(f"Error calculating emotion_words_ratio: {str(e)}")
                features['emotion_words_ratio'] = 0.0
                
            try:
                if words and sentences:
                    avg_word_len = sum(len(word) for word in words) / len(words)
                    avg_sent_len = len(words) / len(sentences)
                    features['complexity_score'] = (avg_word_len * 0.3) + (avg_sent_len * 0.7)
                else:
                    features['complexity_score'] = 0.0
            except Exception as e:
                logger.error(f"Error calculating complexity_score: {str(e)}")
                features['complexity_score'] = 0.0
                
            # Ensure all features are present and of correct type
            for feature in self.feature_names:
                if feature not in features or not isinstance(features[feature], (int, float)):
                    logger.warning(f"Invalid value for {feature}, using default")
                    features[feature] = 0.0
                    
            return features
            
        except Exception as e:
            logger.error(f"Unexpected error in extract_style_features: {str(e)}")
            return default_features
        
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