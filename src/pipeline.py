"""
Main pipeline module
Orchestrates the full end-to-end MBTI analysis flow
"""

import json
import os
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .preprocessing import preprocess_text, chunk_text, clean_mbti_response
from .embedding import SemanticEmbedder, create_semantic_embedding
from .style_embedding import StyleEmbedder, create_style_embedding
from .retrieval import VectorRetriever
from .deduplication import ResponseDeduplicator, deduplicate_responses
from .prompt_builder import PromptBuilder


class MBTIPipeline:
    """Main pipeline for MBTI personality analysis"""
    
    def __init__(self, data_path: str = "mbti_dataset/mbti_responses_800.json"):
        """
        Initialize MBTI pipeline
        
        Args:
            data_path: Path to MBTI dataset
        """
        self.data_path = data_path
        self.semantic_embedder = SemanticEmbedder()
        self.style_embedder = StyleEmbedder()
        self.semantic_retriever = VectorRetriever(embedding_dim=384)
        self.style_retriever = VectorRetriever(embedding_dim=9)  # Style features count
        self.deduplicator = ResponseDeduplicator()
        self.prompt_builder = PromptBuilder()
        
        self.is_initialized = False
    
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the pipeline by loading and processing data
        
        Args:
            force_rebuild: Whether to force rebuild of vector databases
        """
        print("Initializing MBTI Pipeline...")
        
        # Check if vector databases exist
        semantic_db_path = "data/semantic_vectors.json"
        style_db_path = "data/style_vectors.json"
        
        if not force_rebuild and Path(semantic_db_path).exists() and Path(style_db_path).exists():
            print("Loading existing vector databases...")
            self.semantic_retriever.load_from_file(semantic_db_path)
            self.style_retriever.load_from_file(style_db_path)
        else:
            print("Building vector databases from dataset...")
            self._build_vector_databases()
            
            # Save vector databases
            os.makedirs("data", exist_ok=True)
            self.semantic_retriever.save_to_file(semantic_db_path)
            self.style_retriever.save_to_file(style_db_path)
        
        self.is_initialized = True
        print(f"Pipeline initialized successfully!")
        print(f"Semantic DB: {self.semantic_retriever.get_stats()}")
        print(f"Style DB: {self.style_retriever.get_stats()}")
    
    def _build_vector_databases(self):
        """Build vector databases from MBTI dataset"""
        # Load dataset
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Processing {len(data)} MBTI responses...")
        
        for i, response in enumerate(data):
            if i % 100 == 0:
                print(f"Processed {i}/{len(data)} responses...")
            
            # Clean response
            cleaned_response = clean_mbti_response(response)
            text = cleaned_response.get('posts', '') or cleaned_response.get('text', '')
            
            if not text:
                continue
            
            # Preprocess text
            processed_text = preprocess_text(text)
            
            # Create chunks for long texts
            chunks = chunk_text(processed_text, max_chunk_size=512)
            
            for chunk_idx, chunk in enumerate(chunks):
                # Create embeddings
                semantic_emb = self.semantic_embedder.create_embedding(chunk)
                style_emb = self.style_embedder.create_embedding(chunk)
                
                # Prepare metadata
                metadata = {
                    'original_index': i,
                    'chunk_index': chunk_idx,
                    'mbti_type': cleaned_response.get('type', ''),
                    'full_text': text,
                    'chunk_text': chunk
                }
                
                # Add to vector databases
                self.semantic_retriever.add_document(chunk, semantic_emb, metadata)
                self.style_retriever.add_document(chunk, style_emb, metadata)
    
    def analyze_text(
        self,
        query_text: str,
        k: int = 5,
        semantic_weight: float = 0.8,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze text for MBTI personality traits
        
        Args:
            query_text: Text to analyze
            k: Number of similar examples to retrieve
            semantic_weight: Weight for semantic vs style similarity
            
        Returns:
            Analysis results dictionary
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        print(f"Analyzing text: '{query_text[:100]}...'")
        
        # Preprocess query
        processed_query = preprocess_text(query_text)
        
        # Create embeddings for query
        query_semantic_emb = self.semantic_embedder.create_embedding(processed_query)
        query_style_emb = self.style_embedder.create_embedding(processed_query)
        
        # Adjust deduplicator weights based on provided semantic weight
        self.deduplicator.semantic_weight = semantic_weight
        self.deduplicator.style_weight = 1 - semantic_weight

        # Retrieve similar responses
        semantic_results = self.semantic_retriever.search_similar(
            query_semantic_emb, k=k*2, threshold=0.1
        )
        
        style_results = self.style_retriever.search_similar(
            query_style_emb, k=k*2, threshold=0.1
        )
        
        # Combine and process results
        all_results = []
        
        # Add semantic results with embeddings
        for doc_id, similarity, metadata in semantic_results:
            result = metadata.copy()
            result['semantic_similarity'] = similarity
            result['semantic_embedding'] = self.semantic_retriever.vectors[doc_id]
            result['style_embedding'] = self.style_retriever.vectors[doc_id] if doc_id < len(self.style_retriever.vectors) else None
            all_results.append(result)
        
        # Add style results
        for doc_id, similarity, metadata in style_results:
            # Check if already added from semantic results
            existing = next((r for r in all_results if r.get('original_index') == metadata.get('original_index') and r.get('chunk_index') == metadata.get('chunk_index')), None)
            if existing:
                existing['style_similarity'] = similarity
            else:
                result = metadata.copy()
                result['style_similarity'] = similarity
                result['semantic_embedding'] = self.semantic_retriever.vectors[doc_id] if doc_id < len(self.semantic_retriever.vectors) else None
                result['style_embedding'] = self.style_retriever.vectors[doc_id]
                all_results.append(result)
        
        # Deduplicate and rank results
        final_results = deduplicate_responses(
            all_results, query_semantic_emb, query_style_emb, self.deduplicator
        )[:k]
        
        # Build prompt
        prompt = self.prompt_builder.build_analysis_prompt(
            processed_query, final_results
        )

        llm_response = None
        parsed_response = None
        if use_llm:
            try:
                from app.generate.gemini import generate_gemini_response

                llm_response = generate_gemini_response(
                    f"{self.prompt_builder.system_prompt}\n\n{prompt}"
                )
                parsed_response = self.prompt_builder.extract_mbti_from_response(
                    llm_response
                )
            except Exception as e:
                llm_response = f"LLM call failed: {e}"

        return {
            'query': query_text,
            'processed_query': processed_query,
            'similar_responses': final_results,
            'prompt': prompt,
            'llm_response': llm_response,
            'parsed_response': parsed_response,
            'query_embeddings': {
                'semantic': query_semantic_emb.tolist(),
                'style': query_style_emb.tolist(),
            },
        }
    
    def compare_texts(self, text1: str, text2: str) -> Dict[str, Any]:
        """
        Compare two texts for personality similarity
        
        Args:
            text1: First text
            text2: Second text
            
        Returns:
            Comparison results dictionary
        """
        if not self.is_initialized:
            raise ValueError("Pipeline not initialized. Call initialize() first.")
        
        # Analyze both texts
        analysis1 = self.analyze_text(text1)
        analysis2 = self.analyze_text(text2)
        
        # Build comparison prompt
        prompt = self.prompt_builder.build_comparison_prompt(text1, text2)
        
        return {
            'text1': text1,
            'text2': text2,
            'analysis1': analysis1,
            'analysis2': analysis2,
            'comparison_prompt': prompt
        }
    
    def get_pipeline_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        if not self.is_initialized:
            return {'status': 'not_initialized'}
        
        return {
            'status': 'initialized',
            'semantic_db': self.semantic_retriever.get_stats(),
            'style_db': self.style_retriever.get_stats(),
            'deduplicator_weights': {
                'semantic': self.deduplicator.semantic_weight,
                'style': self.deduplicator.style_weight
            }
        }
