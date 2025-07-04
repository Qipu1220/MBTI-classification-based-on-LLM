"""
MBTI Personality Analysis App - Streamlit Interface
"""

import streamlit as st
import json
import os
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import pandas as pd
from pathlib import Path
import time
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Import local modules
from src.pipeline import MBTIPipeline
from src.embedding import SemanticEmbedder
from src.style_embedding import StyleEmbedder

# Page config
st.set_page_config(
    page_title="MBTI Personality Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# MBTI Questions
MBTI_QUESTIONS = [
    "Trong một buổi tiệc đông người, bạn thường làm gì ngay khi bước vào và điều đó ảnh hưởng thế nào đến mức năng lượng của bạn?",
    "Kể về lần bạn cần nạp lại năng lượng sau một ngày căng thẳng; bạn chọn hoạt động gì và tại sao?",
    "Hãy mô tả cách bạn quan sát một cảnh thiên nhiên quen thuộc – bạn chú ý chi tiết cụ thể nào đầu tiên?",
    "Khi học một kỹ năng mới, bạn ưu tiên trải nghiệm trực tiếp hay hình dung các khả năng trước? Giải thích.",
    "Hãy kể lại lần bạn giải quyết vấn đề bằng cách dựa vào thực tế trước mắt thay vì giả thuyết xa.",
    "Đưa ra ví dụ khi bạn tin tưởng trực giác hơn số liệu; điều gì khiến bạn quyết định như vậy?",
    "Khi phải phán đoán một ý tưởng gây tranh cãi, bạn đánh giá dựa trên nguyên tắc logic nào?",
    "Mô tả một tình huống bạn đặt ưu tiên cho cảm xúc con người thay vì lập luận lý trí thuần túy.",
    "Bạn định nghĩa \"công bằng\" như thế nào trong một cuộc tranh luận quan trọng?",
    "Hãy kể lần bạn phải thông báo quyết định khó khăn cho người khác và cách bạn cân bằng giữa khách quan và đồng cảm.",
    "Buổi sáng đi làm, bạn có lịch trình cố định hay thích linh hoạt tùy hứng? Minh họa bằng ví dụ.",
    "Khi nhận dự án dài hạn, bạn lập kế hoạch chi tiết đến mức nào trước khi bắt đầu?",
    "Kể về lần bạn thay đổi kế hoạch vào phút chót – điều gì thôi thúc bạn linh hoạt?",
    "Bạn cảm thấy thế nào khi không kịp hoàn thành mục tiêu đúng hạn, và bạn phản ứng ra sao?",
    "Mô tả \"một ngày hoàn hảo\" theo bạn, từ khi thức dậy đến lúc đi ngủ; hãy chú ý mức độ cấu trúc vs. ngẫu hứng trong ngày đó."
]

def initialize_session_state():
    """Initialize session state variables"""
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None
    if 'pipeline_initialized' not in st.session_state:
        st.session_state.pipeline_initialized = False
    if 'analysis_history' not in st.session_state:
        st.session_state.analysis_history = []
    if 'survey_responses' not in st.session_state:
        st.session_state.survey_responses = {}
    if 'survey_completed' not in st.session_state:
        st.session_state.survey_completed = False
    if 'mbti_result' not in st.session_state:
        st.session_state.mbti_result = None

def initialize_pipeline():
    """
    Initialize the MBTI pipeline
    
    Returns:
        tuple: (success: bool, message: str)
    """
    try:
        # Check if pipeline is already initialized
        if st.session_state.get('pipeline_initialized', False):
            return True, "Pipeline đã được khởi tạo trước đó."
            
        # Check if dataset exists - use absolute path
        base_dir = Path(__file__).parent
        dataset_path = base_dir / "mbti_dataset" / "mbti_responses_800.json"
        dataset_path = dataset_path.resolve()  # Convert to absolute path
        
        if not dataset_path.exists():
            error_msg = f"Không tìm thấy tập dữ liệu MBTI tại {dataset_path}. Vui lòng đảm bảo file tồn tại."
            logger.error(error_msg)
            return False, error_msg
        
        logger.info(f"Đang khởi tạo pipeline với dữ liệu từ: {dataset_path}")
        
        # Initialize pipeline
        try:
            pipeline = MBTIPipeline(
                data_dir=str(dataset_path.parent),
                semantic_model_name="all-MiniLM-L6-v2",
                device=None,  # Auto-detect device
                enable_caching=True
            )
            
            # Explicitly call initialize to ensure everything is set up
            if not pipeline.initialize():
                error_msg = "Không thể khởi tạo pipeline. Vui lòng kiểm tra log để biết thêm chi tiết."
                logger.error(error_msg)
                return False, error_msg
                
            st.session_state.pipeline = pipeline
            st.session_state.pipeline_initialized = True
            
            # Get pipeline stats
            try:
                stats = pipeline.get_pipeline_stats()
                st.session_state.pipeline_stats = stats
                logger.info(f"Pipeline stats: {stats}")
            except Exception as stats_error:
                logger.warning(f"Không thể lấy thống kê pipeline: {str(stats_error)}")
            
            logger.info("Khởi tạo pipeline thành công")
            return True, "Khởi tạo pipeline thành công!"
            
        except Exception as init_error:
            error_msg = f"Lỗi khi khởi tạo pipeline: {str(init_error)}"
            logger.error(error_msg, exc_info=True)
            return False, error_msg
        
    except Exception as e:
        error_msg = f"Lỗi không xác định khi khởi tạo pipeline: {str(e)}"
        logger.error(error_msg, exc_info=True)
        return False, error_msg

def format_responses_for_analysis(responses: Dict[str, str]) -> str:
    """Format survey responses for MBTI analysis to match the dataset format"""
    formatted_responses = []
    for i, (question, answer) in enumerate(responses.items(), 1):
        # Format each question-answer pair to match the dataset format
        formatted = f"Q{i}: {question}\nAns: {answer}"
        formatted_responses.append(formatted)
    
    # Join all responses with newlines to match the dataset format
    return "\n".join(formatted_responses)

def analyze_mbti_responses(responses: Dict[str, str]) -> Dict:
    """Analyze MBTI survey responses using the pipeline"""
    if not st.session_state.pipeline_initialized:
        st.error("Pipeline not initialized. Please initialize it first.")
        return None
    
    # Format responses to match the dataset format
    formatted_responses = format_responses_for_analysis(responses)
    
    try:
        with st.spinner("Analyzing your MBTI personality..."):
            # First try with LLM enabled
            result = st.session_state.pipeline.analyze_text(
                formatted_responses, 
                k=10,  # Get more examples for better analysis
                semantic_weight=0.7,
                use_llm=True  # Always try to use LLM for better accuracy
            )
            
            # If no similar responses found, try with a different approach
            if not result.get('top_similar') or len(result.get('top_similar', [])) == 0:
                st.warning("No similar responses found in database. Using advanced analysis...")
                
                # Try with different parameters
                result = st.session_state.pipeline.analyze_text(
                    formatted_responses,
                    k=20,  # Try with more results
                    semantic_weight=0.5,  # Balance between semantic and style
                    use_llm=True
                )
            
            # Map the result keys to match what the UI expects
            mapped_result = {
                'analysis': {
                    'similar_responses': result.get('top_similar', []),
                    'summary': result.get('analysis', {}).get('summary', 'No analysis available'),
                    'prompt': result.get('prompt', 'No prompt available'),
                    'formatted_responses': formatted_responses,  # Include the formatted responses for reference
                    'llm_used': True,  # Indicate that LLM was used
                    'predicted_type': None  # Will be set by predict_mbti_type
                },
                'predicted_type': predict_mbti_type({
                    **result,
                    'analysis': {
                        **result.get('analysis', {}),
                        'predicted_type': result.get('analysis', {}).get('predicted_type')
                    }
                })
            }
            
            # Ensure we have a predicted type
            if not mapped_result['predicted_type'] or mapped_result['predicted_type'] == 'Unknown':
                # If still no prediction, try to extract from LLM response
                if 'analysis' in result and 'llm_response' in result['analysis']:
                    llm_response = result['analysis']['llm_response']
                    if isinstance(llm_response, str):
                        # Try to extract MBTI type from LLM response
                        import re
                        mbti_match = re.search(r'\b([IE][NS][TF][PJ])\b', llm_response.upper())
                        if mbti_match:
                            mapped_result['predicted_type'] = mbti_match.group(1)
            
            # Add any additional fields that might be needed
            if 'query_text' in result:
                mapped_result['query_text'] = result['query_text']
            if 'processed_text' in result:
                mapped_result['processed_text'] = result['processed_text']
            
            return mapped_result
            
    except Exception as e:
        st.error(f"Analysis failed: {e}")
        st.error(f"Error details: {str(e)}")
        import traceback
        st.error(f"Traceback: {traceback.format_exc()}")
        
        # Return a fallback result with the formatted responses
        return {
            'analysis': {
                'similar_responses': [],
                'summary': 'Analysis failed. Using fallback prediction.',
                'prompt': formatted_responses,
                'formatted_responses': formatted_responses,
                'llm_used': False
            },
            'predicted_type': predict_mbti_type({
                'query_text': ' '.join(responses.values()),
                'processed_text': formatted_responses,
                'analysis': {}
            })
        }
        return None

def predict_mbti_type(analysis_result: Dict) -> str:
    """Predict MBTI type based on analysis results"""
    if not analysis_result:
        return "Unknown"
    
    # First, try to get prediction from LLM analysis if available
    if analysis_result.get('analysis') and isinstance(analysis_result['analysis'], dict):
        # Check if LLM provided a prediction
        llm_response = analysis_result['analysis'].get('predicted_type')
        if llm_response and llm_response != "Unknown":
            return llm_response
    
    # Fall back to similar responses if available
    similar_responses = analysis_result.get('similar_responses') or analysis_result.get('top_similar', [])
    
    # Count MBTI types from similar responses
    type_counter = {}
    for response in similar_responses:
        mbti_type = response.get('mbti_type')
        if mbti_type and mbti_type != 'Unknown':
            score = response.get('final_score', 1.0)  # Default score of 1.0 if not provided
            type_counter[mbti_type] = type_counter.get(mbti_type, 0) + score
    
    # If we have similar responses, return the most common type
    if type_counter:
        return max(type_counter.items(), key=lambda x: x[1])[0]
    
    # If no similar responses, check if we can use LLM for prediction
    if 'prompt' in analysis_result and st.session_state.get('pipeline_initialized', False):
        try:
            # Call LLM directly with the prompt
            llm_response = st.session_state.pipeline._call_llm(analysis_result['prompt'])
            if isinstance(llm_response, dict):
                predicted = llm_response.get('predicted_type')
                if predicted and predicted != "Unknown":
                    return predicted
        except Exception as e:
            st.error(f"Error calling LLM for prediction: {str(e)}")
    
    # Final fallback: analyze the text directly for MBTI traits
    text_to_analyze = analysis_result.get('query_text', analysis_result.get('processed_text', ''))
    if text_to_analyze:
        # Simple heuristic: count MBTI trait indicators in text
        traits = {
            'E': text_to_analyze.upper().count('EXTROVERT') + text_to_analyze.upper().count('EXTRAVERT'),
            'I': text_to_analyze.upper().count('INTROVERT'),
            'S': text_to_analyze.upper().count('SENSING') + text_to_analyze.upper().count('SENSATION'),
            'N': text_to_analyze.upper().count('INTUITION') + text_to_analyze.upper().count('INTUITIVE'),
            'T': text_to_analyze.upper().count('THINKING') + text_to_analyze.upper().count('THOUGHT'),
            'F': text_to_analyze.upper().count('FEELING') + text_to_analyze.upper().count('FEELINGS') + text_to_analyze.upper().count('FEEL'),
            'J': text_to_analyze.upper().count('JUDGING') + text_to_analyze.upper().count('JUDGMENT'),
            'P': text_to_analyze.upper().count('PERCEIVING') + text_to_analyze.upper().count('PERCEPTION')
        }
        
        # Build MBTI type from most common traits
        mbti = ''
        mbti += 'E' if traits['E'] > traits['I'] else 'I'
        mbti += 'N' if traits['N'] > traits['S'] else 'S'
        mbti += 'T' if traits['T'] > traits['F'] else 'F'
        mbti += 'J' if traits['J'] > traits['P'] else 'P'
        
        # Validate the MBTI type
        valid_mbti = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP',
                     'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']
        
        if mbti in valid_mbti:
            return mbti
    
    # If all else fails, return a default MBTI type
    return "ENFP"  # A common neutral type

def survey_interface():
    """MBTI Survey interface"""
    st.header("🧠 MBTI Personality Survey")
    st.markdown("*Trả lời 15 câu hỏi sau để khám phá tính cách MBTI của bạn*")
    
    if not st.session_state.survey_completed:
        st.info("Hãy trả lời tất cả 15 câu hỏi một cách chi tiết và chân thật để có kết quả chính xác nhất.")
        
        # Survey form
        with st.form("mbti_survey"):
            responses = {}
            
            for i, question in enumerate(MBTI_QUESTIONS, 1):
                st.subheader(f"Câu hỏi {i}")
                st.write(question)
                
                # Get existing response if available
                existing_response = st.session_state.survey_responses.get(f"q{i}", "")
                
                response = st.text_area(
                    f"Câu trả lời {i}:",
                    value=existing_response,
                    height=100,
                    key=f"response_{i}",
                    placeholder="Hãy trả lời chi tiết và cụ thể..."
                )
                responses[question] = response
            
            submitted = st.form_submit_button("Phân tích tính cách MBTI", type="primary")
            
            if submitted:
                # Validate responses
                empty_responses = [i+1 for i, (_, response) in enumerate(responses.items()) if not response.strip()]
                
                if empty_responses:
                    st.error(f"Vui lòng trả lời đầy đủ các câu hỏi: {', '.join(map(str, empty_responses))}")
                else:
                    try:
                        # Save responses in both formats for backward compatibility
                        st.session_state.survey_responses = {
                            f"q{i+1}": response for i, (_, response) in enumerate(responses.items())
                        }
                        
                        # Also save the question-answer pairs for analysis
                        st.session_state.qa_responses = responses
                        
                        # Analyze responses
                        analysis_result = analyze_mbti_responses(responses)
                        
                        if analysis_result:
                            # Predict MBTI type
                            predicted_mbti = predict_mbti_type(analysis_result)
                            
                            # Save results
                            st.session_state.mbti_result = {
                                'predicted_type': predicted_mbti,
                                'analysis': analysis_result,
                                'formatted_responses': analysis_result.get('analysis', {}).get('formatted_responses', ''),
                                'timestamp': time.time()
                            }
                            
                            # Debug: Show the formatted responses being sent for analysis
                            st.session_state.debug_formatted_responses = analysis_result.get('analysis', {}).get('formatted_responses', '')
                            
                    except Exception as e:
                        st.error(f"An error occurred while processing your responses: {str(e)}")
                        import traceback
                        st.error(f"Error details: {traceback.format_exc()}")
                        st.stop()
                    
                    st.session_state.survey_completed = True
                    st.rerun()
    
    else:
        # Show results
        st.success("🎉 Khảo sát đã hoàn thành!")
        
        if st.session_state.mbti_result:
            result = st.session_state.mbti_result
            
            # Display predicted MBTI type
            st.subheader("🎯 Kết quả phân tích tính cách")
            
            col1, col2 = st.columns([1, 2])
            
            # Initialize default values
            predicted_type = "Unknown"
            similar_responses = []
            
            # Safely get values from result
            if result and isinstance(result, dict):
                predicted_type = result.get('predicted_type', 'Unknown')
                if 'analysis' in result and isinstance(result['analysis'], dict):
                    similar_responses = result['analysis'].get('similar_responses', [])
            
            with col1:
                st.metric(
                    "Loại tính cách MBTI của bạn:",
                    predicted_type,
                    help="Kết quả dựa trên phân tích các phản hồi của bạn"
                )
            
            with col2:
                # Show confidence based on top matches if we have similar responses
                if similar_responses:
                    # Use final_score if available, otherwise use hybrid_score, default to 0
                    top_scores = [
                        r.get('final_score', r.get('hybrid_score', 0)) 
                        for r in similar_responses[:3]
                    ]
                    avg_confidence = sum(top_scores) / len(top_scores) if top_scores else 0
                    st.metric(
                        "Độ tin cậy:",
                        f"{avg_confidence:.1%}",
                        help="Mức độ phù hợp với các mẫu tương tự trong dữ liệu"
                    )
            
            # Show basic information about the analysis
            st.write("Phân tích đã hoàn thành dựa trên các phản hồi của bạn.")
            
            # Show confidence based on top matches if we have similar responses
            if similar_responses:
                # Use final_score if available, otherwise use hybrid_score, default to 0
                top_scores = [
                    r.get('final_score', r.get('hybrid_score', 0)) 
                    for r in similar_responses[:3]
                ]
                avg_confidence = sum(top_scores) / len(top_scores) if top_scores else 0
                st.write(f"**Độ tin cậy phân tích:** {avg_confidence:.1%}")
            
            # Show generated analysis prompt if available
            prompt = result.get('analysis', {}).get('prompt')
            if prompt and prompt != 'No prompt available':
                with st.expander("🤖 Prompt phân tích được tạo"):
                    st.code(prompt, language="markdown")
            
            # Restart survey option
            st.subheader("🔄 Làm lại khảo sát")
            if st.button("Làm lại khảo sát từ đầu", type="secondary"):
                # Reset survey state
                st.session_state.survey_responses = {}
                st.session_state.survey_completed = False
                st.session_state.mbti_result = None
                st.rerun()

def analyze_text_interface():
    """Text analysis interface"""
    st.header("🔍 Text Analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter text to analyze:",
        placeholder="Type or paste the text you want to analyze for MBTI personality traits...",
        height=150
    )
    
    # Analysis parameters
    col1, col2 = st.columns(2)
    with col1:
        k_similar = st.slider("Number of similar examples", 1, 10, 5)
    with col2:
        semantic_weight = st.slider("Semantic vs Style weight", 0.0, 1.0, 0.7)
    
    if st.button("Analyze Text", type="primary"):
        if not text_input.strip():
            st.warning("Please enter some text to analyze.")
            return
        
        if not st.session_state.pipeline_initialized:
            st.error("Pipeline not initialized. Please initialize it first.")
            return
        
        try:
            with st.spinner("Analyzing text..."):
                result = st.session_state.pipeline.analyze_text(
                    text_input, 
                    k=k_similar, 
                    semantic_weight=semantic_weight
                )
            
            # Display results
            st.success("Analysis completed!")
            
            # Store in history
            st.session_state.analysis_history.append({
                'timestamp': time.time(),
                'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                'result': result
            })
            
            # Show similar responses
            if result['similar_responses']:
                st.subheader("📊 Similar Responses Found")
                
                for i, response in enumerate(result['similar_responses'], 1):
                    with st.expander(f"Match {i}: MBTI {response.get('mbti_type', 'Unknown')} (Score: {response.get('hybrid_score', 0):.3f})"):
                        st.write("**Text:**", response.get('chunk_text', '')[:200] + "...")
                        if 'semantic_similarity' in response:
                            st.write("**Semantic Similarity:**", f"{response['semantic_similarity']:.3f}")
                        if 'style_similarity' in response:
                            st.write("**Style Similarity:**", f"{response['style_similarity']:.3f}")
            
            # Show generated prompt
            with st.expander("🤖 Generated Analysis Prompt"):
                st.code(result['prompt'], language="markdown")
            
        except Exception as e:
            st.error(f"Analysis failed: {e}")

def comparison_interface():
    """Text comparison interface"""
    st.header("⚖️ Text Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Text 1")
        text1 = st.text_area(
            "First text:",
            key="text1",
            placeholder="Enter the first text...",
            height=150
        )
    
    with col2:
        st.subheader("Text 2")
        text2 = st.text_area(
            "Second text:",
            key="text2",
            placeholder="Enter the second text...",
            height=150
        )
    
    if st.button("Compare Texts", type="primary"):
        if not text1.strip() or not text2.strip():
            st.warning("Please enter both texts to compare.")
            return
        
        if not st.session_state.pipeline_initialized:
            st.error("Pipeline not initialized. Please initialize it first.")
            return
        
        try:
            with st.spinner("Comparing texts..."):
                comparison = st.session_state.pipeline.compare_texts(text1, text2)
            
            st.success("Comparison completed!")
            
            # Show comparison prompt
            with st.expander("🤖 Generated Comparison Prompt"):
                st.code(comparison['comparison_prompt'], language="markdown")
            
            # Show individual analyses
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Text 1 Analysis")
                if comparison['analysis1']['similar_responses']:
                    top_match = comparison['analysis1']['similar_responses'][0]
                    st.info(f"Top match: {top_match.get('mbti_type', 'Unknown')} (Score: {top_match.get('hybrid_score', 0):.3f})")
            
            with col2:
                st.subheader("Text 2 Analysis")
                if comparison['analysis2']['similar_responses']:
                    top_match = comparison['analysis2']['similar_responses'][0]
                    st.info(f"Top match: {top_match.get('mbti_type', 'Unknown')} (Score: {top_match.get('hybrid_score', 0):.3f})")
            
        except Exception as e:
            st.error(f"Comparison failed: {e}")

def history_interface():
    """Analysis history interface"""
    st.header("📝 Analysis History")
    
    # Include survey results in history
    all_history = []
    
    # Add survey result if exists
    if st.session_state.mbti_result:
        all_history.append({
            'type': 'survey',
            'timestamp': st.session_state.mbti_result['timestamp'],
            'result': st.session_state.mbti_result
        })
    
    # Add text analysis history
    for item in st.session_state.analysis_history:
        all_history.append({
            'type': 'text_analysis',
            'timestamp': item['timestamp'],
            'text': item['text'],
            'result': item['result']
        })
    
    if not all_history:
        st.info("No analysis history yet. Complete the MBTI survey or analyze some texts to see them here!")
        return
    
    # Clear history button
    if st.button("Clear History", type="secondary"):
        st.session_state.analysis_history = []
        st.session_state.mbti_result = None
        st.session_state.survey_completed = False
        st.session_state.survey_responses = {}
        st.rerun()
    
    # Sort by timestamp (newest first)
    all_history.sort(key=lambda x: x['timestamp'], reverse=True)
    
    # Display history
    for i, item in enumerate(all_history, 1):
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(item['timestamp']))
        
        if item['type'] == 'survey':
            with st.expander(f"MBTI Survey {i} - {timestamp}"):
                st.write("**Predicted MBTI Type:**", item['result']['predicted_type'])
                if item['result']['analysis']['similar_responses']:
                    st.write("**Top matches:**")
                    for j, response in enumerate(item['result']['analysis']['similar_responses'][:3], 1):
                        st.write(f"{j}. MBTI {response.get('mbti_type', 'Unknown')} (Score: {response.get('hybrid_score', 0):.3f})")
        else:
            with st.expander(f"Text Analysis {i} - {timestamp}"):
                st.write("**Text:**", item['text'])
                
                if item['result']['similar_responses']:
                    st.write("**Top matches:**")
                    for j, response in enumerate(item['result']['similar_responses'][:3], 1):
                        st.write(f"{j}. MBTI {response.get('mbti_type', 'Unknown')} (Score: {response.get('hybrid_score', 0):.3f})")

def settings_interface():
    """Settings and configuration interface"""
    st.header("⚙️ Settings")
    
    # Pipeline status
    st.subheader("Pipeline Status")
    if st.session_state.pipeline_initialized:
        stats = st.session_state.pipeline.get_pipeline_stats()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success("✅ Pipeline Initialized")
            st.json(stats)
        
        with col2:
            if st.button("Reinitialize Pipeline"):
                st.session_state.pipeline_initialized = False
                st.session_state.pipeline = None
                st.rerun()
    else:
        st.warning("❌ Pipeline Not Initialized")
        if st.button("Initialize Pipeline"):
            if initialize_pipeline():
                st.rerun()
    
    # Gemini API status
    st.subheader("Gemini API Configuration")
    try:
        key_manager = APIKeyManager()
        st.success(f"✅ {key_manager.get_key_count()} API keys available")
    except Exception as e:
        st.error(f"❌ Gemini API configuration error: {e}")
        st.info("Please configure your .env file with valid Gemini API keys")
    
    # Survey data management
    st.subheader("Survey Data")
    if st.session_state.survey_responses:
        st.write(f"**Responses saved:** {len(st.session_state.survey_responses)} answers")
        if st.button("Clear Survey Data"):
            st.session_state.survey_responses = {}
            st.session_state.survey_completed = False
            st.session_state.mbti_result = None
            st.success("Survey data cleared!")
            st.rerun()
    else:
        st.info("No survey data available")
    
    # About
    st.subheader("About")
    st.info("""
    This MBTI Personality Analysis app uses a sophisticated pipeline combining:
    - **15-Question Survey** for comprehensive personality assessment
    - **Semantic embeddings** for content understanding
    - **Style embeddings** for writing pattern analysis  
    - **Hybrid retrieval** for finding similar personality examples
    - **Prompt engineering** for structured analysis
    
    The system processes responses through Unicode normalization, chunking, 
    embedding creation, similarity search, and deduplication to provide 
    accurate personality insights based on the Myers-Briggs Type Indicator.
    """)

def main():
    """Main app function"""
    # Initialize session state
    initialize_session_state()
    
    # App header
    st.title("🧠 MBTI Personality Analyzer")
    st.markdown("*Khám phá tính cách của bạn thông qua khảo sát 15 câu hỏi*")
    
    # Initialize pipeline if not already done
    if not st.session_state.pipeline_initialized:
        with st.spinner("Đang khởi tạo hệ thống phân tích..."):
            success, message = initialize_pipeline()
            if not success:
                st.error(f"Lỗi: {message}")
                st.info("Vui lòng kiểm tra lại đường dẫn đến tập dữ liệu và thử lại.")
                return
        
        # Show success message if initialization was successful
        st.success("✅ Hệ thống đã sẵn sàng!")
    
    # Show the survey interface
    survey_interface()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**MBTI Personality Survey**")
    st.sidebar.markdown("Built with Streamlit & Python")

if __name__ == "__main__":
    # Allow running the Streamlit app via `python app.py`
    import os
    import sys
    import subprocess
    if os.getenv("RUNNING_STREAMLIT_APP") != "true":
        os.environ["RUNNING_STREAMLIT_APP"] = "true"
        subprocess.run(["streamlit", "run", sys.argv[0]])
    else:
        main()
