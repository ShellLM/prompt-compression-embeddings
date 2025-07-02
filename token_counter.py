#!/usr/bin/env python3
"""
Token counting utilities for prompt compression analysis.
"""

import re
from typing import Dict

class TokenCounter:
    """Simple token counter using various estimation methods."""
    
    @staticmethod
    def estimate_tokens_simple(text: str) -> int:
        """
        Simple token estimation: ~4 characters per token for English text.
        This is a rough approximation used by OpenAI.
        """
        return len(text) // 4
    
    @staticmethod
    def estimate_tokens_whitespace(text: str) -> int:
        """
        Whitespace-based token estimation.
        Splits on whitespace and punctuation.
        """
        # Split on whitespace and common punctuation
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return len(tokens)
    
    @staticmethod  
    def estimate_tokens_word_based(text: str) -> int:
        """
        Word-based estimation with adjustments for common patterns.
        Generally more accurate than simple character counting.
        """
        # Split into words
        words = text.split()
        
        # Adjust for common tokenization patterns
        token_count = 0
        for word in words:
            # Long words often get split into multiple tokens
            if len(word) > 12:
                token_count += max(2, len(word) // 6)
            elif len(word) > 8:
                token_count += 2
            else:
                token_count += 1
                
        # Add tokens for punctuation and special characters
        punct_count = len(re.findall(r'[^\w\s]', text))
        token_count += punct_count * 0.5  # Punctuation often shares tokens
        
        return int(token_count)
    
    @staticmethod
    def get_all_estimates(text: str) -> Dict[str, int]:
        """Get token estimates using all methods."""
        return {
            'simple_chars': TokenCounter.estimate_tokens_simple(text),
            'whitespace': TokenCounter.estimate_tokens_whitespace(text),
            'word_based': TokenCounter.estimate_tokens_word_based(text),
        }
    
    @staticmethod
    def get_best_estimate(text: str) -> int:
        """Get the most accurate token estimate (word-based method)."""
        return TokenCounter.estimate_tokens_word_based(text)

if __name__ == '__main__':
    # Test the token counter
    test_text = "This is a test prompt with some technical terminology and punctuation!"
    estimates = TokenCounter.get_all_estimates(test_text)
    print(f"Test text: {test_text}")
    print(f"Length: {len(test_text)} characters")
    print("Token estimates:")
    for method, count in estimates.items():
        print(f"  {method}: {count} tokens")
