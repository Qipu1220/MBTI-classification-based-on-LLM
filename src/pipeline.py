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
    
    def __init__(self, data_path: str = None, semantic_model: str = "all-MiniLM-L6-v2"):
        """
        Initialize MBTI pipeline
        
        Args:
            data_path: Path to MBTI dataset. If None, will use default path relative to app.py
            semantic_model: Name of the semantic embedding model to use
        """
        if data_path is None:
            # Default to mbti_dataset/mbti_responses_800.json in the project root
            base_dir = Path(__file__).parent.parent  # Go up from src/ to project root
            self.data_path = str(base_dir / "mbti_dataset" / "mbti_responses_800.json")
        else:
            self.data_path = data_path
            
        # Initialize embedders with configurable models
        self.semantic_embedder = SemanticEmbedder(model_name=semantic_model)
        self.style_embedder = StyleEmbedder()
        
        # Initialize retrievers with appropriate dimensions
        self.semantic_retriever = VectorRetriever(embedding_dim=384)  # Dimension for all-MiniLM-L6-v2
        self.style_retriever = VectorRetriever(embedding_dim=9)  # Style features count
        
        # Initialize other components
        self.deduplicator = ResponseDeduplicator()
        self.prompt_builder = PromptBuilder()
        
        # Initialize state
        self.is_initialized = False
        self.embedding_model = semantic_model
    
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the pipeline by loading and processing data
        
        Args:
            force_rebuild: Whether to force rebuild of vector databases
        """
        print("Initializing MBTI Pipeline...")
        
        # Create data directory in the same directory as the dataset
        data_dir = Path(self.data_path).parent.parent / "data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Define vector database paths
        semantic_db_path = data_dir / "semantic_vectors.json"
        style_db_path = data_dir / "style_vectors.json"
        
        if not force_rebuild and semantic_db_path.exists() and style_db_path.exists():
            print("Loading existing vector databases...")
            try:
                self.semantic_retriever.load_from_file(str(semantic_db_path))
                self.style_retriever.load_from_file(str(style_db_path))
                print("Successfully loaded vector databases.")
            except Exception as e:
                print(f"Error loading vector databases: {e}")
                print("Rebuilding vector databases...")
                force_rebuild = True
        
        if force_rebuild or not semantic_db_path.exists() or not style_db_path.exists():
            print("Building vector databases from dataset...")
            self._build_vector_databases()
            
            # Save vector databases
            try:
                self.semantic_retriever.save_to_file(str(semantic_db_path))
                self.style_retriever.save_to_file(str(style_db_path))
                print(f"Vector databases saved to {data_dir}")
            except Exception as e:
                print(f"Warning: Could not save vector databases: {e}")
        
        self.is_initialized = True
        print(f"Pipeline initialized successfully!")
        print(f"Semantic DB: {self.semantic_retriever.get_stats()}")
        print(f"Style DB: {self.style_retriever.get_stats()}")
    
    def _build_vector_databases(self):
        """Build vector databases from MBTI dataset"""
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Check if data directory is writable
        if not os.access('data', os.W_OK):
            raise PermissionError("Không có quyền ghi vào thư mục 'data'")
        
        # Load dataset
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {self.data_path}")
            
        with open(self.data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Processing {len(data)} MBTI responses...")
        
        for i, response in enumerate(data):
            if i % 100 == 0:
                print(f"Processed {i}/{len(data)} responses...")
            
            try:
                # Clean and extract response
                cleaned_response = clean_mbti_response(response)
                answer_text = cleaned_response.get('answer_input', '')
                
                if not answer_text:
                    print(f"Skipping response {i}: empty answer")
                    continue
                
                # Preprocess text
                processed_text = preprocess_text(answer_text, use_unidecode=False)
                
                # Create 5 chunks
                chunks = chunk_text(processed_text, num_chunks=5)
                
                if not chunks:
                    print(f"Skipping response {i}: failed to create chunks")
                    continue
                
                # Create embeddings for each chunk
                for chunk_idx, chunk in enumerate(chunks):
                    try:
                        # Create semantic embedding
                        semantic_emb = self.semantic_embedder.create_embedding(chunk)
                        
                        # Create style embedding
                        style_features = self._extract_style_features(chunk)
                        style_emb = np.array(style_features)
                        
                        # Prepare metadata
                        metadata = {
                            'original_index': i,
                            'chunk_index': chunk_idx,
                            'mbti_type': cleaned_response.get('mbti', ''),
                            'full_text': processed_text,
                            'chunk_text': chunk,
                            'style_features': style_features
                        }
                        
                        # Add to retrievers
                        self.semantic_retriever.add_item(semantic_emb, metadata)
                        self.style_retriever.add_item(style_emb, metadata)
                        
                    except Exception as e:
                        print(f"Error creating embeddings for chunk {chunk_idx} of response {i}: {str(e)}")
                        continue
            
            except Exception as e:
                print(f"Error processing response {i}: {str(e)}")
                continue
    
    def _extract_style_features(self, text: str) -> List[float]:
        """Extract style features from text"""
        features = []
        
        # Sentence length features
        sentences = re.split(r'[.!?]+', text)
        avg_sentence_len = np.mean([len(s.split()) for s in sentences]) if sentences else 0
        features.append(avg_sentence_len)
        
        # Emoji features (count different types of emojis)
        emoji_pattern = re.compile(r'[\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F300-\U0001F5FF\U00002694-\U00002697\U00002699\U0000269B\U0000269C\U000026A0\U000026A1\U000026A3-\U000026A9\U000026AA\U000026AB\U000026B0-\U000026B1\U000026BD-\U000026BF\U000026C4-\U000026C5\U000026C8\U000026CE\U000026CF\U000026D1\U000026D3\U000026D4\U000026E9-\U000026EA\U000026F0-\U000026F5\U000026F7-\U000026FA\U000026FD\U0001F170-\U0001F171\U0001F17E-\U0001F17F\U0001F18E\U0001F191-\U0001F19A\U0001F201-\U0001F202\U0001F21A\U0001F22F\U0001F232-\U0001F23A\U0001F250\U0001F251\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF\U0001F700-\U0001F77F\U0001F780-\U0001F7FF\U0001F800-\U0001F8FF\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U0001FB00-\U0001FBFF\U0001FCD0-\U0001FCE0\U0001FCE1-\U0001FCFF\U0001FD00-\U0001FDFF\U0001FE00-\U0001FEFF\U0001FF00-\U0001FF60\U0001FF61-\U0001FF65]')
        emojis = emoji_pattern.findall(text)
        features.append(len(emojis))
        
        # POS ratio features
        from nltk import pos_tag, word_tokenize
        try:
            tokens = word_tokenize(text)
            tags = pos_tag(tokens)
            
            # Count different POS tags
            pos_counts = {
                'NOUN': len([t for t in tags if t[1].startswith('NN')]),
                'VERB': len([t for t in tags if t[1].startswith('VB')]),
                'ADJ': len([t for t in tags if t[1].startswith('JJ')]),
                'ADV': len([t for t in tags if t[1].startswith('RB')])
            }
            
            total_words = len(tokens)
            if total_words > 0:
                features.extend([
                    pos_counts['NOUN'] / total_words,
                    pos_counts['VERB'] / total_words,
                    pos_counts['ADJ'] / total_words,
                    pos_counts['ADV'] / total_words
                ])
            else:
                features.extend([0, 0, 0, 0])
        except Exception as e:
            print(f"Error extracting POS features: {str(e)}")
            features.extend([0, 0, 0, 0])
        
        # Add more features as needed
        # features.extend([other_features...])
        
        return features
    
    def analyze_text(
        self,
        query_text: str,
        k: int = 5,
        semantic_weight: float = 0.8,
        use_llm: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze text for MBTI personality traits using both semantic and style embeddings
        
        Args:
            query_text: Text to analyze
            k: Number of similar examples to retrieve
            semantic_weight: Weight for semantic vs style similarity (0.0 to 1.0)
            use_llm: Whether to use LLM for final analysis
            
        Returns:
            Analysis results dictionary containing:
            - query_text: Original input text
            - processed_text: Preprocessed text
            - semantic_embedding: Semantic embedding vector
            - style_embedding: Style embedding vector
            - similar_documents: List of similar documents with scores
            - analysis: MBTI analysis results (if use_llm=True)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
            
        if not query_text or not isinstance(query_text, str):
            raise ValueError("query_text must be a non-empty string")
            
        if not 0 <= semantic_weight <= 1:
            raise ValueError("semantic_weight must be between 0.0 and 1.0")
        
        print(f"Analyzing text with {self.embedding_model} (semantic weight: {semantic_weight:.2f})\nText: '{query_text[:100]}...'\n")
        
        # Preprocess query text
        processed_text = preprocess_text(query_text, use_unidecode=False)
        print(f"Preprocessed text: '{processed_text[:100]}...'\n")
        
        # Create embeddings
        print("Creating embeddings...")
        query_sem_emb = self.semantic_embedder.create_embedding(processed_text)
        query_style_emb = self.style_embedder.create_embedding(processed_text)
        
        # Get initial semantic results (wider net)
        semantic_results = self.semantic_retriever.search(
            query_sem_emb, 
            k=min(20, len(self.semantic_retriever)),  # Get more results for re-ranking
            include_metadata=True
        )
        print(f"Retrieved {len(semantic_results)} semantic similar documents")
        
        # If no semantic results, try with style only
        if not semantic_results and semantic_weight >= 1.0:
            semantic_weight = 0.5  # Fallback to equal weighting
            
        # Re-rank with style similarity
        re_ranked_results = []
        style_weight = 1.0 - semantic_weight
        
        # Extract style features for query
        query_style_features = self._extract_style_features(processed_text)
        query_style_emb = np.array(query_style_features)
        print("Extracted style features for query\n")
        
        # Calculate style similarity and combine scores
        for item in semantic_results:
            # Get style embedding from metadata
            doc_style_features = item['metadata']['style_features']
            doc_style_emb = np.array(doc_style_features)
            
            # Calculate style similarity
            style_score = np.dot(query_style_emb, doc_style_emb) / (
                np.linalg.norm(query_style_emb) * np.linalg.norm(doc_style_emb)
            )
            
            # Normalize scores to [0,1]
            sem_score = (item['score'] + 1) / 2  # Assuming scores are in [-1,1]
            style_score = (style_score + 1) / 2  # Normalize to [0,1]
            
            # Calculate final score with alpha=0.8
            final_score = 0.8 * sem_score + 0.2 * style_score
            
            re_ranked_results.append({
                'doc_id': item['metadata']['original_index'],
                'chunk_id': item['metadata']['chunk_index'],
                'final_score': final_score,
                'semantic_score': sem_score,
                'style_score': style_score,
                'text': item['metadata']['chunk_text'],
                'full_text': item['metadata']['full_text'],
                'mbti_type': item['metadata']['mbti_type']
            })
        
        # Sort by final score and take top-k
        re_ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        top_results = re_ranked_results[:k]
        print(f"Selected top {k} results after re-ranking\n")
        
        # Prepare prompt for LLM if needed
        if use_llm:
            prompt = self.prompt_builder.build_analysis_prompt(
                query_text=processed_text,
                similar_responses=top_results,
                context={
                    'top_k': k,
                    'semantic_weight': semantic_weight
                }
            )
            
            # Call LLM API
            analysis = self._call_llm(prompt)
        else:
            analysis = self._analyze_without_llm(top_results)
        
        return {
            'query_text': query_text,
            'processed_text': processed_text,
            'top_similar': top_results,
            'analysis': analysis,
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
