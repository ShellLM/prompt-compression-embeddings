#!/usr/bin/env python3
"""
Quick demo of the prompt compression system.
"""

import asyncio
import sys
from prompt_compression_test import PromptCompressor, EmbeddingComparer, TokenCounter

async def demo_compression():
    """Run a quick demo with a sample prompt."""
    
    sample_prompt = """I need to create a comprehensive marketing strategy for a new eco-friendly product line that includes sustainable packaging, renewable energy sources in manufacturing, and a focus on millennial and Gen Z consumers who are environmentally conscious. The strategy should cover digital marketing channels, social media campaigns, influencer partnerships, and traditional advertising methods. Please provide detailed recommendations for budget allocation, timeline, key performance indicators, and success metrics. Also include competitor analysis and market positioning strategies."""
    
    print("=== PROMPT COMPRESSION DEMO ===\n")
    print(f"Original prompt ({len(sample_prompt)} chars, ~{TokenCounter.get_best_estimate(sample_prompt)} tokens):")
    print(f'"{sample_prompt}"\n')
    
    compressor = PromptCompressor("openrouter/google/gemini-2.5-flash-preview-05-20")
    embedder = EmbeddingComparer()
    
    strategies = ["summarize", "extract_core", "minimal"]
    
    print("Testing compression strategies:")
    
    # Get original embedding
    print("Computing original embedding...")
    original_embedding = embedder.get_embedding(sample_prompt)
    
    for strategy in strategies:
        print(f"\n--- {strategy.upper()} ---")
        
        try:
            # Compress
            compressed = compressor.compress_prompt(sample_prompt, strategy)
            
            # Get compressed embedding
            compressed_embedding = embedder.get_embedding(compressed)
            
            # Calculate metrics
            char_ratio = len(compressed) / len(sample_prompt)
            token_ratio = TokenCounter.get_best_estimate(compressed) / TokenCounter.get_best_estimate(sample_prompt)
            similarity = embedder.cosine_similarity(original_embedding, compressed_embedding)
            
            print(f"Compressed ({len(compressed)} chars, ~{TokenCounter.get_best_estimate(compressed)} tokens):")
            print(f'"{compressed}"')
            print(f"Character compression: {char_ratio:.2f} ({(1-char_ratio)*100:.1f}% reduction)")
            print(f"Token compression: {token_ratio:.2f} ({(1-token_ratio)*100:.1f}% reduction)")
            print(f"Semantic similarity: {similarity:.3f}")
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == '__main__':
    asyncio.run(demo_compression())
