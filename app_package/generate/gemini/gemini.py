"""Gemini API interface for generating MBTI analysis"""

from typing import Optional
import os
import google.generativeai as genai

from .api_key_manager import APIKeyManager
from .reset_api_key import retry_with_backoff


def _configure_genai(api_key: str):
    """Configure the generative AI client."""
    genai.configure(api_key=api_key)


def generate_gemini_response(prompt: str, model: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 1024) -> str:
    """Generate text from Gemini model using the given prompt."""
    key_manager = APIKeyManager()
    api_key = key_manager.get_current_key()
    _configure_genai(api_key)

    model_name = model or os.getenv("GEMINI_MODEL", "gemini-pro")

    def _call():
        gen_model = genai.GenerativeModel(model_name)
        response = gen_model.generate_content(
            prompt,
            generation_config={"temperature": temperature, "max_output_tokens": max_tokens},
        )
        return response.text

    # Use retry_with_backoff to handle errors and rotate keys if needed
    return retry_with_backoff(key_manager, _call)
