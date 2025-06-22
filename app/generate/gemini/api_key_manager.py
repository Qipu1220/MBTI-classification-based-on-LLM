"""
API Key Manager for Gemini
Wrapper around APIKeyManager with rotation logic
"""

import os
import random
from typing import List, Optional


class APIKeyManager:
    """Manages multiple Gemini API keys with rotation"""
    
    def __init__(self):
        self.api_keys = self._load_api_keys()
        self.current_index = 0
        
    def _load_api_keys(self) -> List[str]:
        """Load API keys from environment variable"""
        api_list = os.getenv('GEMINI_API_LIST', '')
        if not api_list:
            raise ValueError("GEMINI_API_LIST environment variable not set")
        
        keys = [key.strip() for key in api_list.split(',') if key.strip()]
        if not keys:
            raise ValueError("No valid API keys found in GEMINI_API_LIST")
        
        return keys
    
    def get_current_key(self) -> str:
        """Get the current API key"""
        if not self.api_keys:
            raise ValueError("No API keys available")
        return self.api_keys[self.current_index]
    
    def rotate_key(self) -> str:
        """Rotate to the next API key"""
        if len(self.api_keys) <= 1:
            return self.get_current_key()
        
        self.current_index = (self.current_index + 1) % len(self.api_keys)
        return self.get_current_key()
    
    def get_random_key(self) -> str:
        """Get a random API key"""
        if not self.api_keys:
            raise ValueError("No API keys available")
        return random.choice(self.api_keys)
    
    def get_key_count(self) -> int:
        """Get the number of available API keys"""
        return len(self.api_keys)
    
    def reset_to_first(self) -> str:
        """Reset to the first API key"""
        self.current_index = 0
        return self.get_current_key()