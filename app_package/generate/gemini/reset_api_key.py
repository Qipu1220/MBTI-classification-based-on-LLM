"""
API key rotation logic for Gemini
Handles automatic key rotation on rate limits or errors
"""

import time
import logging
from typing import Optional
from .api_key_manager import APIKeyManager


logger = logging.getLogger(__name__)


def reset_api_key(key_manager: APIKeyManager, error_msg: str = "") -> str:
    """
    Reset API key after encountering an error
    
    Args:
        key_manager: APIKeyManager instance
        error_msg: Error message that triggered the reset
        
    Returns:
        New API key
    """
    logger.warning(f"Resetting API key due to error: {error_msg}")
    
    # Rotate to next key
    new_key = key_manager.rotate_key()
    
    # Add delay to avoid hitting rate limits immediately
    time.sleep(1)
    
    logger.info(f"Switched to new API key (key {key_manager.current_index + 1}/{key_manager.get_key_count()})")
    
    return new_key


def handle_api_error(key_manager: APIKeyManager, error: Exception) -> Optional[str]:
    """
    Handle API errors and determine if key rotation is needed
    
    Args:
        key_manager: APIKeyManager instance
        error: Exception that occurred
        
    Returns:
        New API key if rotation occurred, None otherwise
    """
    error_msg = str(error).lower()
    
    # Check if error indicates rate limiting or quota issues
    rate_limit_indicators = [
        'rate limit',
        'quota exceeded',
        'too many requests',
        'resource exhausted',
        'retry after'
    ]
    
    if any(indicator in error_msg for indicator in rate_limit_indicators):
        logger.warning(f"Rate limit detected: {error}")
        return reset_api_key(key_manager, str(error))
    
    # Check for authentication errors
    auth_error_indicators = [
        'unauthorized',
        'invalid api key',
        'authentication failed',
        'forbidden'
    ]
    
    if any(indicator in error_msg for indicator in auth_error_indicators):
        logger.error(f"Authentication error: {error}")
        return reset_api_key(key_manager, str(error))
    
    # For other errors, don't rotate key
    logger.error(f"Non-recoverable error: {error}")
    return None


def retry_with_backoff(key_manager: APIKeyManager, func, max_retries: int = 3, base_delay: float = 1.0):
    """
    Retry function with exponential backoff and key rotation
    
    Args:
        key_manager: APIKeyManager instance
        func: Function to retry
        max_retries: Maximum number of retry attempts
        base_delay: Base delay between retries
        
    Returns:
        Function result
        
    Raises:
        Last exception if all retries failed
    """
    last_exception = None
    
    for attempt in range(max_retries + 1):
        try:
            return func()
        except Exception as e:
            last_exception = e
            
            if attempt < max_retries:
                # Try rotating key
                new_key = handle_api_error(key_manager, e)
                
                # Calculate delay with exponential backoff
                delay = base_delay * (2 ** attempt)
                logger.info(f"Retry attempt {attempt + 1}/{max_retries} after {delay}s delay")
                time.sleep(delay)
            else:
                logger.error(f"All retry attempts failed")
    
    raise last_exception