"""
Deduplication module
Hybrid retrieval ranking and deduplication of responses
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Set
from collections import defaultdict
import hashlib


class ResponseDeduplicator:
    """Handles deduplication and ranking of retrieved responses"""
    
    def __init__(self, semantic_weight: float = 0.7, style_weight: float = 0.3):
        """
        Initialize deduplicator
        
        Args:
            semantic_weight: Weight for semantic similarity
            style_weight: Weight for style similarity
        """
        self.semantic_weight = semantic_weight
        self.style_weight = style_weight
    
    def deduplicate_by_content(self, responses: List[Dict[str, Any]], 
                              similarity_threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        Remove duplicate responses based on content similarity
        
        Args:
            responses: List of response dictionaries
            similarity_threshold: Threshold for considering responses as duplicates
            
        Returns:
            Deduplicated list of responses
        """
        if not responses:
            return []
        
        # Group responses by content hash for exact duplicates
        content_groups = defaultdict(list)
        for response in responses:
            content = response.get('text', '')
            content_hash = hashlib.md5(content.encode()).hexdigest()
            content_groups[content_hash].append(response)
        
        # Keep only one response per exact content group
        unique_responses = []
        for group in content_groups.values():
            # Keep the response with highest similarity/score if available
            best_response = max(
                group,
                key=lambda x: x.get('hybrid_score', x.get('semantic_similarity', 0)),
            )
            unique_responses.append(best_response)
        
        return unique_responses
    
    def hybrid_ranking(self, query_semantic_emb: np.ndarray, query_style_emb: np.ndarray,
                      responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Rank responses using hybrid semantic + style similarity
        
        Args:
            query_semantic_emb: Query semantic embedding
            query_style_emb: Query style embedding
            responses: List of response dictionaries with embeddings
            
        Returns:
            Ranked list of responses
        """
        scored_responses = []
        
        for response in responses:
            semantic_emb = response.get('semantic_embedding')
            style_emb = response.get('style_embedding')
            
            # Calculate semantic similarity
            semantic_sim = 0.0
            if semantic_emb is not None:
                semantic_sim = self._cosine_similarity(query_semantic_emb, semantic_emb)
            
            # Calculate style similarity
            style_sim = 0.0
            if style_emb is not None:
                style_sim = self._cosine_similarity(query_style_emb, style_emb)
            
            # Compute hybrid score
            hybrid_score = (self.semantic_weight * semantic_sim + 
                           self.style_weight * style_sim)
            
            response_copy = response.copy()
            response_copy['hybrid_score'] = hybrid_score
            response_copy['semantic_similarity'] = semantic_sim
            response_copy['style_similarity'] = style_sim
            
            scored_responses.append(response_copy)
        
        # Sort by hybrid score descending
        scored_responses.sort(key=lambda x: x['hybrid_score'], reverse=True)
        
        return scored_responses
    
    def diversify_responses(self, responses: List[Dict[str, Any]], 
                           diversity_threshold: float = 0.8, max_responses: int = 5) -> List[Dict[str, Any]]:
        """
        Diversify responses to avoid too similar results
        
        Args:
            responses: Ranked list of responses
            diversity_threshold: Minimum diversity threshold
            max_responses: Maximum number of responses to return
            
        Returns:
            Diversified list of responses
        """
        if not responses:
            return []
        
        diversified = [responses[0]]  # Always include the top response
        
        for response in responses[1:]:
            if len(diversified) >= max_responses:
                break
            
            # Check diversity against already selected responses
            is_diverse = True
            for selected in diversified:
                # Compare semantic embeddings
                sem_emb1 = response.get('semantic_embedding')
                sem_emb2 = selected.get('semantic_embedding')
                
                if sem_emb1 is not None and sem_emb2 is not None:
                    similarity = self._cosine_similarity(sem_emb1, sem_emb2)
                    if similarity > diversity_threshold:
                        is_diverse = False
                        break
            
            if is_diverse:
                diversified.append(response)
        
        return diversified
    
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


def deduplicate_responses(responses: List[Dict[str, Any]], 
                         query_semantic_emb: np.ndarray = None,
                         query_style_emb: np.ndarray = None,
                         deduplicator: ResponseDeduplicator = None) -> List[Dict[str, Any]]:
    """
    Deduplicate and rank responses
    
    Args:
        responses: List of response dictionaries
        query_semantic_emb: Query semantic embedding for ranking
        query_style_emb: Query style embedding for ranking
        deduplicator: Optional deduplicator instance
        
    Returns:
        Deduplicated and ranked responses
    """
    if deduplicator is None:
        deduplicator = ResponseDeduplicator()
    
    # First remove exact duplicates
    unique_responses = deduplicator.deduplicate_by_content(responses)
    
    # Then apply hybrid ranking if embeddings are provided
    if query_semantic_emb is not None and query_style_emb is not None:
        ranked_responses = deduplicator.hybrid_ranking(
            query_semantic_emb, query_style_emb, unique_responses
        )
        # Finally diversify results
        final_responses = deduplicator.diversify_responses(ranked_responses)
        return final_responses
    
    return unique_responses
