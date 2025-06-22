"""
Vector database interface and retrieval module
Handles vector storage and similarity search
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import json
from pathlib import Path


class VectorRetriever:
    """Vector database interface for similarity search"""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize vector retriever
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        self.embedding_dim = embedding_dim
        self.vectors = []
        self.metadata = []
        self.index_map = {}
    
    def add_document(self, text: str, embedding: np.ndarray, metadata: Dict[str, Any]) -> int:
        """
        Add document with embedding to the database
        
        Args:
            text: Original text
            embedding: Document embedding
            metadata: Document metadata
            
        Returns:
            Document index
        """
        doc_id = len(self.vectors)
        
        self.vectors.append(embedding)
        self.metadata.append({
            'text': text,
            'doc_id': doc_id,
            **metadata
        })
        
        return doc_id
    
    def add_documents_batch(self, texts: List[str], embeddings: List[np.ndarray], 
                           metadata_list: List[Dict[str, Any]]) -> List[int]:
        """
        Add multiple documents in batch
        
        Args:
            texts: List of texts
            embeddings: List of embeddings
            metadata_list: List of metadata dictionaries
            
        Returns:
            List of document indices
        """
        doc_ids = []
        for text, embedding, metadata in zip(texts, embeddings, metadata_list):
            doc_id = self.add_document(text, embedding, metadata)
            doc_ids.append(doc_id)
        return doc_ids
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5, 
                      threshold: float = 0.0) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search for similar documents
        
        Args:
            query_embedding: Query embedding
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (doc_id, similarity, metadata) tuples
        """
        if not self.vectors:
            return []
        
        similarities = []
        for i, doc_embedding in enumerate(self.vectors):
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            if similarity >= threshold:
                similarities.append((i, similarity, self.metadata[i]))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:k]
    
    def search_by_text(self, query_text: str, embedding_func, k: int = 5, 
                      threshold: float = 0.0) -> List[Tuple[int, float, Dict[str, Any]]]:
        """
        Search using text query
        
        Args:
            query_text: Query text
            embedding_func: Function to create embedding from text
            k: Number of results to return
            threshold: Minimum similarity threshold
            
        Returns:
            List of (doc_id, similarity, metadata) tuples
        """
        query_embedding = embedding_func(query_text)
        return self.search_similar(query_embedding, k, threshold)
    
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """
        Get document by ID
        
        Args:
            doc_id: Document ID
            
        Returns:
            Document metadata or None
        """
        if 0 <= doc_id < len(self.metadata):
            return self.metadata[doc_id]
        return None
    
    def save_to_file(self, filepath: str):
        """
        Save vector database to file
        
        Args:
            filepath: Path to save file
        """
        data = {
            'vectors': [vec.tolist() for vec in self.vectors],
            'metadata': self.metadata,
            'embedding_dim': self.embedding_dim
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_from_file(self, filepath: str):
        """
        Load vector database from file
        
        Args:
            filepath: Path to load file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vectors = [np.array(vec) for vec in data['vectors']]
        self.metadata = data['metadata']
        self.embedding_dim = data.get('embedding_dim', 384)
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors"""
        if vec1.size == 0 or vec2.size == 0:
            return 0.0
        
        # Normalize vectors
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(vec1_norm, vec2_norm)
        return float(similarity)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        return {
            'total_documents': len(self.vectors),
            'embedding_dimension': self.embedding_dim,
            'memory_usage_mb': sum(vec.nbytes for vec in self.vectors) / (1024 * 1024)
        }