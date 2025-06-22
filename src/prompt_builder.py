"""
Prompt builder module
Structured prompt assembly for MBTI analysis
"""

from typing import List, Dict, Any, Optional
import json


class PromptBuilder:
    """Builds structured prompts for MBTI personality analysis"""
    
    def __init__(self):
        """Initialize prompt builder"""
        self.system_prompt = self._get_system_prompt()
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for MBTI analysis"""
        return """You are an expert MBTI personality analyst. Your task is to analyze text and provide insights about the author's personality type based on their writing style, word choices, and expressed thoughts.

When analyzing text, consider:
1. Extraversion (E) vs Introversion (I): Energy source and social orientation
2. Sensing (S) vs Intuition (N): Information processing preference  
3. Thinking (T) vs Feeling (F): Decision-making approach
4. Judging (J) vs Perceiving (P): Lifestyle and structure preference

Provide your analysis in a structured format with:
- Primary personality indicators
- Supporting evidence from the text
- Confidence level for each dimension
- Overall MBTI type assessment"""
    
    def build_analysis_prompt(self, query_text: str, similar_responses: List[Dict[str, Any]], 
                             context: Optional[Dict[str, Any]] = None) -> str:
        """
        Build prompt for MBTI analysis with retrieved examples
        
        Args:
            query_text: Text to analyze
            similar_responses: List of similar MBTI responses for context
            context: Optional additional context
            
        Returns:
            Structured prompt string
        """
        prompt_parts = []
        
        # Add query text
        prompt_parts.append("## Text to Analyze:")
        prompt_parts.append(f'"{query_text}"')
        prompt_parts.append("")
        
        # Add similar examples if available
        if similar_responses:
            prompt_parts.append("## Similar Examples from Database:")
            for i, response in enumerate(similar_responses[:3], 1):
                text = response.get('text', '')
                mbti_type = response.get('type', 'Unknown')
                confidence = response.get('hybrid_score', 0)
                
                prompt_parts.append(f"### Example {i} (MBTI: {mbti_type}, Relevance: {confidence:.2f}):")
                prompt_parts.append(f'"{text[:200]}..."')
                prompt_parts.append("")
        
        # Add analysis request
        prompt_parts.append("## Analysis Request:")
        prompt_parts.append("Based on the text above and the similar examples provided, analyze the personality traits and provide:")
        prompt_parts.append("1. **Extraversion (E) vs Introversion (I)** - with evidence and confidence (0-1)")
        prompt_parts.append("2. **Sensing (S) vs Intuition (N)** - with evidence and confidence (0-1)")
        prompt_parts.append("3. **Thinking (T) vs Feeling (F)** - with evidence and confidence (0-1)")
        prompt_parts.append("4. **Judging (J) vs Perceiving (P)** - with evidence and confidence (0-1)")
        prompt_parts.append("5. **Overall MBTI Type** - most likely type with overall confidence")
        prompt_parts.append("6. **Key Indicators** - specific words/phrases that support your analysis")
        
        # Add context if provided
        if context:
            prompt_parts.append("")
            prompt_parts.append("## Additional Context:")
            for key, value in context.items():
                prompt_parts.append(f"- {key}: {value}")
        
        return "\n".join(prompt_parts)
    
    def build_comparison_prompt(self, text1: str, text2: str, 
                               analysis1: Optional[str] = None, 
                               analysis2: Optional[str] = None) -> str:
        """
        Build prompt for comparing two texts
        
        Args:
            text1: First text
            text2: Second text
            analysis1: Optional analysis of first text
            analysis2: Optional analysis of second text
            
        Returns:
            Comparison prompt string
        """
        prompt_parts = []
        
        prompt_parts.append("## Text Comparison Analysis")
        prompt_parts.append("")
        
        prompt_parts.append("### Text 1:")
        prompt_parts.append(f'"{text1}"')
        if analysis1:
            prompt_parts.append(f"Previous Analysis: {analysis1}")
        prompt_parts.append("")
        
        prompt_parts.append("### Text 2:")
        prompt_parts.append(f'"{text2}"')
        if analysis2:
            prompt_parts.append(f"Previous Analysis: {analysis2}")
        prompt_parts.append("")
        
        prompt_parts.append("### Comparison Request:")
        prompt_parts.append("Compare these two texts and analyze:")
        prompt_parts.append("1. **Personality Similarities** - shared traits and indicators")
        prompt_parts.append("2. **Personality Differences** - contrasting traits and evidence")
        prompt_parts.append("3. **Writing Style Comparison** - tone, structure, word choice")
        prompt_parts.append("4. **MBTI Compatibility** - how these personality types might interact")
        prompt_parts.append("5. **Confidence Assessment** - reliability of the comparison")
        
        return "\n".join(prompt_parts)
    
    def build_summary_prompt(self, analyses: List[str], texts: List[str]) -> str:
        """
        Build prompt for summarizing multiple analyses
        
        Args:
            analyses: List of analysis results
            texts: List of corresponding texts
            
        Returns:
            Summary prompt string
        """
        prompt_parts = []
        
        prompt_parts.append("## Multiple Text Analysis Summary")
        prompt_parts.append("")
        
        for i, (text, analysis) in enumerate(zip(texts, analyses), 1):
            prompt_parts.append(f"### Analysis {i}:")
            prompt_parts.append(f"Text: \"{text[:100]}...\"")
            prompt_parts.append(f"Analysis: {analysis}")
            prompt_parts.append("")
        
        prompt_parts.append("### Summary Request:")
        prompt_parts.append("Based on the analyses above, provide:")
        prompt_parts.append("1. **Overall Personality Pattern** - common traits across texts")
        prompt_parts.append("2. **Consistency Analysis** - how consistent the personality indicators are")
        prompt_parts.append("3. **Dominant MBTI Type** - most likely overall personality type")
        prompt_parts.append("4. **Confidence Assessment** - reliability of the overall assessment")
        prompt_parts.append("5. **Recommendations** - suggestions for further analysis if needed")
        
        return "\n".join(prompt_parts)
    
    def extract_mbti_from_response(self, response: str) -> Dict[str, Any]:
        """
        Extract structured MBTI data from LLM response
        
        Args:
            response: LLM response text
            
        Returns:
            Structured MBTI analysis dictionary
        """
        # This is a simple extraction - in practice you'd want more sophisticated parsing
        result = {
            'dimensions': {},
            'overall_type': None,
            'confidence': 0.0,
            'key_indicators': [],
            'raw_response': response
        }
        
        # Try to extract MBTI type (simple pattern matching)
        import re
        mbti_pattern = r'\b([EINS][SNFT][TFJ][JP])\b'
        matches = re.findall(mbti_pattern, response.upper())
        if matches:
            result['overall_type'] = matches[0]
        
        # Extract confidence scores (looking for patterns like "confidence: 0.8")
        confidence_pattern = r'confidence[:\s]+([0-9]*\.?[0-9]+)'
        conf_matches = re.findall(confidence_pattern, response.lower())
        if conf_matches:
            try:
                result['confidence'] = float(conf_matches[0])
            except ValueError:
                pass
        
        return result