"""
Main pipeline module
Orchestrates the full end-to-end MBTI analysis flow with performance optimizations
"""

import json
import os
import re
import time
import torch
import logging
import numpy as np
from functools import lru_cache
from typing import List, Dict, Any, Optional, Tuple, Union, Callable
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('mbti_analysis.log')
    ]
)
logger = logging.getLogger(__name__)

try:
    from .preprocessing import preprocess_text, chunk_text, clean_mbti_response
    from .embedding import SemanticEmbedder, create_semantic_embedding
    from .style_embedding import StyleEmbedder, create_style_embedding
    from .retrieval import VectorRetriever
    from .deduplication import ResponseDeduplicator, deduplicate_responses
    from .prompt_builder import PromptBuilder
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    raise


class MBTIPipeline:
    """Main pipeline for MBTI personality analysis"""
    
    def __init__(self, data_dir: str, semantic_model_name: str = "all-MiniLM-L6-v2", 
                 device: Optional[str] = None, enable_caching: bool = True):
        """
        Initialize MBTI analysis pipeline with performance optimizations
        
        Args:
            data_dir: Directory containing MBTI dataset files
            semantic_model_name: Name of the semantic embedding model to use
            device: Device to run the model on ('cuda' or 'cpu'). If None, will auto-detect.
            enable_caching: Whether to enable caching for embeddings and intermediate results
        """
        start_time = time.time()
        self.data_dir = Path(data_dir)
        self.enable_caching = enable_caching
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # Store model name for reference/logging
        self.embedding_model = semantic_model_name
        # Define MBTI types list early to avoid attribute errors during initialization
        self.mbti_types = [
            'INTJ', 'INTP', 'ENTJ', 'ENTP',
            'INFJ', 'INFP', 'ENFJ', 'ENFP',
            'ISTJ', 'ISFJ', 'ESTJ', 'ESFJ',
            'ISTP', 'ISFP', 'ESTP', 'ESFP'
        ]
        # Initialize empty vector databases dict; will be populated later
        self.vector_dbs: Dict[str, Dict[str, Any]] = {}
        
        # Initialize pipeline state
        self.is_initialized = False
        
        # Initialize components
        self.semantic_retriever = None
        self.style_retriever = None
        self.deduplicator = None
        self.prompt_builder = None
        
        # Configure logging
        self.logger = logging.getLogger("MBTIPipeline")
        self.logger.setLevel(logging.INFO)
        
        # Add file handler if not already added
        if not any(isinstance(h, logging.FileHandler) for h in self.logger.handlers):
            file_handler = logging.FileHandler('mbti_analysis.log')
            file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"Initializing MBTIPipeline on device: {self.device}")
        
        # Initialize components with error handling
        try:
            # Warm up GPU if available
            if 'cuda' in str(self.device):
                self._warm_up_gpu()
                
            # Initialize components
            self._initialize_components(semantic_model_name)
            
            # Vector databases will be built in `initialize()` when explicitly called
            init_time = time.time() - start_time
            self.logger.info(f"Core components initialized in {init_time:.2f} seconds. Call `initialize()` to build vector databases.")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize pipeline: {str(e)}", exc_info=True)
            raise
    
    def _warm_up_gpu(self):
        """Warm up the GPU with a small computation."""
        try:
            if 'cuda' in str(self.device):
                # Small tensor operation to initialize CUDA context
                x = torch.randn(10, 10).to(self.device)
                y = torch.randn(10, 10).to(self.device)
                _ = torch.matmul(x, y)
                self.logger.debug("GPU warmup completed")
        except Exception as e:
            self.logger.warning(f"GPU warmup failed: {str(e)}")
    
    def _initialize_components(self, semantic_model_name: str):
        """Initialize all pipeline components with error handling."""
        try:
            # Initialize embedders with error handling
            self.logger.info(f"Initializing semantic embedder with model: {semantic_model_name}")
            try:
                self.semantic_embedder = SemanticEmbedder(
                    model_name=semantic_model_name,
                    device=self.device
                )
            except Exception as e:
                raise RuntimeError(f"Failed to initialize semantic embedder: {str(e)}")
            
            self.logger.info("Initializing style embedder")
            try:
                self.style_embedder = StyleEmbedder()
            except Exception as e:
                raise RuntimeError(f"Failed to initialize style embedder: {str(e)}")
            
            # Initialize other components
            self.semantic_retriever = VectorRetriever()
            self.style_retriever = VectorRetriever()
            self.deduplicator = ResponseDeduplicator()
            self.prompt_builder = PromptBuilder()
            
            # Initialize caches
            self._init_caches()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize components: {str(e)}", exc_info=True)
            raise
    
    def _init_caches(self) -> None:
        """Initialize caching mechanisms with automatic memory management."""
        self._embedding_cache = {}
        self._style_cache = {}
        self._similarity_cache = {}
        
        # Configure cache sizes based on available memory
        self.max_cache_size = 1000  # Default cache size
        try:
            if 'cuda' in str(self.device):
                # Adjust cache size based on available GPU memory
                total_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
                self.max_cache_size = min(5000, int(total_mem * 100))  # ~100MB per GB of VRAM
                self.logger.debug(f"GPU detected, setting cache size to {self.max_cache_size}")
            else:
                # For CPU, use a more conservative cache size
                self.max_cache_size = 1000
                self.logger.debug("Using CPU, setting default cache size")
                
            self.logger.info(f"Initialized caches with max size: {self.max_cache_size}")
            
        except Exception as e:
            self.logger.warning(f"Failed to detect GPU, using default cache size. Error: {str(e)}")
    
    @lru_cache(maxsize=1000)
    def _get_cached_embedding(self, text: str) -> np.ndarray:
        """
        Get or compute semantic embedding with caching.
        
        Args:
            text: Input text to get embedding for
            
        Returns:
            np.ndarray: The embedding vector
            
        Raises:
            RuntimeError: If embedding generation fails
        """
        try:
            # Check cache first
            if text in self._embedding_cache:
                return self._embedding_cache[text]
                
            # Generate new embedding
            embedding = self.semantic_embedder.embed_text(text)
            
            # Update cache if enabled and not full
            if self.enable_caching and len(self._embedding_cache) < self.max_cache_size:
                self._embedding_cache[text] = embedding
                
            return embedding
            
        except Exception as e:
            error_msg = f"Failed to get embedding for text: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e

    
    def initialize(self, force_rebuild: bool = False):
        """
        Initialize the pipeline by loading and processing data
        
        Args:
            force_rebuild: Whether to force rebuild of vector databases
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            if self.is_initialized and not force_rebuild:
                self.logger.info("Pipeline is already initialized")
                return True
                
            self.logger.info("Initializing MBTI Pipeline...")
            
            # Create data directory in the same directory as the dataset
            data_dir = Path(self.data_dir).parent / "data"
            os.makedirs(data_dir, exist_ok=True)
            
            # Define vector database path
            semantic_db_path = data_dir / "semantic_vectors.json"
            
            # Initialize retrievers if not already done
            if self.semantic_retriever is None:
                self.semantic_retriever = VectorRetriever()
            
            if not force_rebuild and semantic_db_path.exists():
                self.logger.info("Loading existing vector databases...")
                try:
                    self.semantic_retriever.load_from_file(str(semantic_db_path))
                    self.logger.info("Successfully loaded vector databases.")
                except Exception as e:
                    self.logger.warning(f"Error loading vector databases: {e}")
                    self.logger.info("Rebuilding vector databases...")
                    force_rebuild = True
            
            if force_rebuild or not semantic_db_path.exists():
                self.logger.info("Building vector databases from dataset...")
                self._build_vector_databases()
                
                # Save vector databases
                try:
                    self.semantic_retriever.save_to_file(str(semantic_db_path))
                    self.logger.info(f"Vector databases saved to {data_dir}")
                except Exception as e:
                    self.logger.warning(f"Could not save vector databases: {e}")
            
            self.is_initialized = True
            self.logger.info("Pipeline initialized successfully")
            return True
            
        except Exception as e:
            self.last_error = str(e)
            self.logger.error(f"Failed to initialize pipeline: {self.last_error}", exc_info=True)
            self.is_initialized = False
            raise
    
    def _build_vector_databases(self):
        """Build vector databases from MBTI dataset"""
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        # Check if data directory is writable
        if not os.access('data', os.W_OK):
            raise PermissionError("Không có quyền ghi vào thư mục 'data'")
        
        # Expect data_dir to be a directory that contains one or more *.json dataset files
        data_path = Path(self.data_dir)
        if not data_path.is_dir():
            raise NotADirectoryError(f"data_dir phải là thư mục chứa các file .json, nhận được: {data_path}")
        
        json_files = list(data_path.glob('*.json'))
        if not json_files:
            raise FileNotFoundError(f"Không tìm thấy file .json trong {data_path}")
        
        print(f"Found {len(json_files)} json files in dataset directory")
        
        for file_idx, file_path in enumerate(json_files, 1):
            print(f"[{file_idx}/{len(json_files)}] Loading {file_path.name} ...")
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    records = json.load(f)
            except Exception as e:
                print(f"Could not parse {file_path}: {e}")
                continue
            
            for i, response in enumerate(records):
                if i % 100 == 0:
                    print(f"Processed {i}/{len(records)} responses...")
                
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
                            
                            # Prepare metadata (style features optional)
                            metadata = {
                                'original_index': i,
                                'chunk_index': chunk_idx,
                                'mbti_type': cleaned_response.get('mbti', ''),
                                'full_text': processed_text,
                                'chunk_text': chunk,
                            }
                            
                            # Add to semantic retriever only
                            self.semantic_retriever.add_item(semantic_emb, metadata)
                            
                        except Exception as e:
                            print(f"Error creating embeddings for chunk {chunk_idx} of response {i}: {str(e)}")
                            continue
                
                except Exception as e:
                    print(f"Error processing response {i}: {str(e)}")
                    continue
    
    def _extract_style_features(self, text: str) -> List[float]:
        """
        Extract style features from text using the StyleEmbedder
        
        Args:
            text: Input text
            
        Returns:
            List of 9 style features (all zeros if extraction fails)
        """
        try:
            # Use the StyleEmbedder to get features
            features_dict = self.style_embedder.extract_style_features(text)
            
            # Ensure we have all expected features in the correct order
            expected_features = [
                'avg_sentence_length', 'avg_word_length', 'punctuation_ratio',
                'question_ratio', 'exclamation_ratio', 'caps_ratio',
                'first_person_ratio', 'emotion_words_ratio', 'complexity_score'
            ]
            
            # Extract features in the expected order, with fallback to 0.0
            features = [float(features_dict.get(feat, 0.0)) for feat in expected_features]
            
            # Ensure we have exactly 9 features
            if len(features) != 9:
                self.logger.warning(f"Expected 9 style features, got {len(features)}")
                features = features[:9] + [0.0] * (9 - len(features))
                
            return features
            
        except Exception as e:
            self.logger.error(f"Unexpected error in _extract_style_features: {str(e)}")
            # Return default values (all zeros) in case of error
            return [0.0] * 9
            return [0.0] * 9
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute cosine similarity between two vectors
        
        Args:
            a: First vector
            b: Second vector
            
        Returns:
            Cosine similarity between a and b
        """
        try:
            # Ensure inputs are numpy arrays
            a = np.asarray(a, dtype=np.float32)
            b = np.asarray(b, dtype=np.float32)
            
            # Handle zero vectors
            norm_a = np.linalg.norm(a)
            norm_b = np.linalg.norm(b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            # Compute cosine similarity
            similarity = np.dot(a, b) / (norm_a * norm_b)
            
            # Ensure the result is within valid range [-1, 1]
            return float(np.clip(similarity, -1.0, 1.0))
            
        except Exception as e:
            logger.error(f"Error in _cosine_similarity: {str(e)}")
            return 0.0
    
    def _build_vector_databases(self) -> None:
        """
        Build vector databases for each MBTI type from the dataset
        """
        try:
            logger.info("Building vector databases...")
            
            # Load and process dataset
            data_files = [f for f in os.listdir(self.data_dir) 
                         if f.endswith('.json') and 'mbti' in f.lower()]
            
            if not data_files:
                raise FileNotFoundError(f"No MBTI dataset files found in {self.data_dir}")
            
            # Process each MBTI type
            for mbti_type in self.mbti_types:
                self.vector_dbs[mbti_type] = {}
                
                # Find files for this MBTI type
                type_files = [f for f in data_files if mbti_type.lower() in f.lower()]
                
                for file in type_files:
                    try:
                        file_path = os.path.join(self.data_dir, file)
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Process each text in the file
                        for i, text in enumerate(data):
                            try:
                                if not text or not isinstance(text, str):
                                    continue
                                    
                                # Create semantic embedding
                                semantic_emb = self.semantic_embedder.create_embedding(text)
                                
                                # Get style features
                                style_features = self._extract_style_features(text)
                                
                                # Combine features
                                combined_emb = np.concatenate([
                                    semantic_emb,
                                    np.array(style_features, dtype=np.float32)
                                ])
                                
                                # Store with a unique key
                                key = f"{mbti_type}_{file}_{i}"
                                self.vector_dbs[mbti_type][key] = combined_emb
                                
                            except Exception as e:
                                logger.warning(f"Error processing text {i} in {file}: {str(e)}")
                                continue
                                
                    except Exception as e:
                        logger.error(f"Error loading {file}: {str(e)}")
                        continue
            
            # Log statistics
            total_vectors = sum(len(vectors) for vectors in self.vector_dbs.values())
            logger.info(f"Built vector databases with {total_vectors} total vectors across {len(self.vector_dbs)} MBTI types")
            
        except Exception as e:
            logger.error(f"Error in _build_vector_databases: {str(e)}")
            raise
    
    def analyze_text(self, text: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Analyze text to predict MBTI type with comprehensive error handling
        
        Args:
            text: Input text to analyze
            top_k: Number of top MBTI types to return (1-16)
            
        Returns:
            Dictionary containing analysis results with error information if any
        """
        # Input validation
        if not text or not isinstance(text, str) or not text.strip():
            error_msg = "Invalid input text"
            logger.warning(error_msg)
            return {
                'error': error_msg,
                'top_matches': [],
                'style_analysis': {},
                'success': False
            }
            
        # Ensure top_k is within valid range
        top_k = max(1, min(top_k, len(self.mbti_types)))
        
        try:
            logger.info(f"Analyzing text (length: {len(text)} chars)")
            
            # Get semantic embedding
            try:
                semantic_emb = self.semantic_embedder.create_embedding(text)
                if not isinstance(semantic_emb, np.ndarray) or semantic_emb.size == 0:
                    raise ValueError("Invalid semantic embedding returned")
            except Exception as e:
                logger.error(f"Error creating semantic embedding: {str(e)}")
                semantic_emb = np.zeros(384)  # Default embedding size for MiniLM
            
            # Get style features
            style_features = self._extract_style_features(text)
            
            # Combine features (simple concatenation)
            try:
                combined_emb = np.concatenate([
                    semantic_emb,
                    np.array(style_features, dtype=np.float32)
                ])
            except Exception as e:
                logger.error(f"Error combining features: {str(e)}")
                combined_emb = np.zeros(384 + 9)  # Default combined size
            
            # Find most similar MBTI types
            similarities = {}
            valid_mbti_types = 0
            
            for mbti_type in self.mbti_types:
                try:
                    if mbti_type not in self.vector_dbs or not self.vector_dbs[mbti_type]:
                        logger.warning(f"No vectors found for MBTI type: {mbti_type}")
                        similarities[mbti_type] = 0.0
                        continue
                        
                    # Get average similarity to examples of this type
                    type_similarities = []
                    for emb in self.vector_dbs[mbti_type].values():
                        try:
                            sim = self._cosine_similarity(combined_emb, emb)
                            if not np.isnan(sim):
                                type_similarities.append(sim)
                        except Exception as e:
                            logger.warning(f"Error calculating similarity for {mbti_type}: {str(e)}")
                    
                    if type_similarities:
                        similarities[mbti_type] = float(np.mean(type_similarities))
                        valid_mbti_types += 1
                    else:
                        similarities[mbti_type] = 0.0
                        
                except Exception as e:
                    logger.error(f"Error processing MBTI type {mbti_type}: {str(e)}")
                    similarities[mbti_type] = 0.0
            
            # If no valid similarities found, return error
            if valid_mbti_types == 0:
                error_msg = "No valid MBTI types found for comparison"
                logger.error(error_msg)
                return {
                    'error': error_msg,
                    'top_matches': [],
                    'style_analysis': {},
                    'success': False
                }
            
            # Sort by similarity
            sorted_types = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
            
            # Get top k types with validation
            top_types = []
            for mbti_type, similarity in sorted_types[:top_k]:
                try:
                    top_types.append({
                        'type': str(mbti_type),
                        'similarity': float(similarity)
                    })
                except Exception as e:
                    logger.warning(f"Error formatting result for {mbti_type}: {str(e)}")
            
            # Prepare style analysis with validation
            style_analysis = {}
            feature_names = [
                'avg_sentence_length', 'avg_word_length', 'punctuation_ratio',
                'question_ratio', 'exclamation_ratio', 'caps_ratio',
                'first_person_ratio', 'emotion_words_ratio', 'complexity_score'
            ]
            
            for i, name in enumerate(feature_names):
                try:
                    style_analysis[name] = float(style_features[i]) if i < len(style_features) else 0.0
                except (IndexError, ValueError, TypeError):
                    style_analysis[name] = 0.0
            
            # Prepare result with validation
            result = {
                'top_matches': top_types,
                'style_analysis': style_analysis,
                'semantic_embedding': semantic_emb.tolist() if isinstance(semantic_emb, np.ndarray) else [0.0] * 384,
                'style_embedding': [float(f) for f in style_features] if style_features else [0.0] * 9,
                'success': True
            }
            
            logger.info(f"Analysis completed. Top match: {top_types[0]['type'] if top_types else 'None'}")
            return result
            
        except Exception as e:
            error_msg = f"Unexpected error in analyze_text: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return {
                'error': error_msg,
                'top_matches': [],
                'style_analysis': {},
                'success': False
            }
    
    def analyze_text(
        self,
        query_text: str,
        k: int = 5,
        semantic_weight: float = 0.8,
        use_llm: bool = False,
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
            'llm_response': analysis,  # Placeholder until LLM integration
            'parsed_response': analysis,
            'query_embeddings': {
                'semantic': query_sem_emb.tolist(),
                'style': query_style_emb.tolist(),
            },
            'embedding_model': self.embedding_model
        }
    
    def _analyze_without_llm(self, top_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback analysis method when no LLM is available."""
        try:
            type_counter = {}
            for item in top_results:
                mbti_type = item.get('mbti_type', 'Unknown')
                type_counter[mbti_type] = type_counter.get(mbti_type, 0) + item.get('final_score', 0)
            if not type_counter:
                return {'summary': 'No similar responses found'}
            predicted_type = max(type_counter, key=type_counter.get)
            return {
                'summary': f'Predicted MBTI type: {predicted_type}',
                'details': type_counter
            }
        except Exception as e:
            self.logger.error(f"_analyze_without_llm failed: {str(e)}")
            return {'summary': 'Analysis error', 'error': str(e)}

    def _call_llm(self, prompt: str) -> Dict[str, Any]:
        """Stub for LLM call. Currently returns placeholder response."""
        self.logger.warning("_call_llm is not implemented. Returning placeholder response.")
        return {'response': 'LLM functionality not implemented', 'prompt': prompt}
         
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
        
        # Get semantic DB stats
        semantic_stats = self.semantic_retriever.get_stats() if hasattr(self.semantic_retriever, 'get_stats') else {'total_documents': 0}
        
        # Initialize style DB stats
        style_stats = {
            'total_documents': 0,
            'dimensions': 0
        }
        
        # If style retriever exists and has get_stats method, use it
        if hasattr(self, 'style_retriever') and hasattr(self.style_retriever, 'get_stats'):
            style_stats = self.style_retriever.get_stats()
        
        return {
            'status': 'initialized',
            'semantic_db': semantic_stats,
            'style_db': style_stats,
            'deduplicator_weights': {
                'semantic': getattr(self.deduplicator, 'semantic_weight', 0.0) if hasattr(self, 'deduplicator') else 0.0,
                'style': getattr(self.deduplicator, 'style_weight', 0.0) if hasattr(self, 'deduplicator') else 0.0
            }
        }
