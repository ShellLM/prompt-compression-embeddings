#!/usr/bin/env python3
"""
Improved Prompt Compression Test

This script tests prompt compression with proper methodology:
1. Compress original prompt -> get compression ratio
2. Decompress the compressed version back to expanded form
3. Compare embeddings of original vs decompressed to measure information loss
4. Test multiple compression strategies and models
"""

import sqlite3
import json
import asyncio
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import subprocess
import tempfile
import os
import re

@dataclass
class TestCase:
    """Represents a single test case with original prompt and metadata."""
    id: str
    model: str
    prompt: str
    response: str
    datetime_utc: str
    prompt_length: int
    response_length: int

@dataclass
class CompressionResult:
    """Results from a compression/decompression cycle."""
    original_prompt: str
    compressed_prompt: str
    decompressed_prompt: str
    compression_strategy: str
    compression_model: str
    decompression_model: str
    original_length: int
    compressed_length: int
    decompressed_length: int
    compression_ratio: float
    decompression_ratio: float
    original_tokens: int
    compressed_tokens: int
    decompressed_tokens: int
    token_compression_ratio: float
    token_decompression_ratio: float
    original_embedding: List[float]
    decompressed_embedding: List[float]
    similarity_score: float

class TokenCounter:
    """Simple token counter using various estimation methods."""
    
    @staticmethod
    def estimate_tokens_simple(text: str) -> int:
        """Simple token estimation: ~4 characters per token for English text."""
        return len(text) // 4
    
    @staticmethod
    def estimate_tokens_whitespace(text: str) -> int:
        """Whitespace-based token estimation."""
        tokens = re.findall(r'\b\w+\b|[^\w\s]', text)
        return len(tokens)
    
    @staticmethod  
    def estimate_tokens_word_based(text: str) -> int:
        """Word-based estimation with adjustments for common patterns."""
        words = text.split()
        
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
    def get_best_estimate(text: str) -> int:
        """Get the most accurate token estimate (word-based method)."""
        return TokenCounter.estimate_tokens_word_based(text)

class PromptCompressor:
    """Handles compression and decompression using LLM."""
    
    def __init__(self, compression_model: str = "gemini-flash", decompression_model: str = "gemini-flash"):
        self.compression_model = compression_model
        self.decompression_model = decompression_model
        
    def compress_prompt(self, prompt: str, strategy: str) -> str:
        """Compress a prompt using the specified strategy."""
        strategies = {
            "ultra_minimal": "Compress this prompt to the absolute minimum words needed while preserving ALL key information. Use abbreviations, remove redundant words, keep only essential context:",
            "structured_compress": "Compress this prompt into a structured format (bullets, key-value pairs, etc.) that captures all information in minimal space:",
            "keyword_dense": "Extract and compress this prompt into dense keyword clusters that preserve all meaning and relationships:",
            "telegram_style": "Compress this prompt like a telegram - every word costs money, but preserve all essential information:",
            "code_like": "Compress this prompt into a code-like structure (pseudo-code, structured notation) that preserves all logic and information:"
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown compression strategy: {strategy}")
            
        compression_instruction = strategies[strategy]
        full_prompt = f"{compression_instruction}\n\n{prompt}"
        
        # Use llm CLI to get compressed version
        result = subprocess.run([
            'llm', '-m', self.compression_model, 
            full_prompt
        ], capture_output=True, text=True, check=True)
        
        return result.stdout.strip()
    
    def decompress_prompt(self, compressed_prompt: str, strategy: str) -> str:
        """Decompress a compressed prompt back to full form."""
        decompression_instruction = f"""
        This is a compressed prompt that was created using the '{strategy}' compression strategy.
        Please expand it back to a full, natural, detailed prompt that captures all the original information and intent.
        Make it complete and well-structured, filling in natural language while preserving all the compressed information.
        
        Compressed prompt: {compressed_prompt}
        """
        
        # Use llm CLI to get decompressed version
        result = subprocess.run([
            'llm', '-m', self.decompression_model, 
            decompression_instruction
        ], capture_output=True, text=True, check=True)
        
        return result.stdout.strip()

class EmbeddingComparer:
    """Handles embedding generation and similarity comparison."""
    
    def __init__(self, embedding_model: str = "jina-v4"):
        self.embedding_model = embedding_model
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using llm CLI."""
        import json
        
        # Use shell to pipe text to llm embed, suppress debug output
        cmd = f'echo {json.dumps(text)} | llm embed -m {self.embedding_model} 2>/dev/null'
        result = subprocess.run(['sh', '-c', cmd], capture_output=True, text=True, check=True)
        
        # Extract just the JSON array part (the last line should be the JSON)
        output_lines = result.stdout.strip().split('\n')
        json_line = output_lines[-1]  # The JSON array should be the last line
        
        # Parse the JSON array output
        embedding_data = json.loads(json_line)
        return embedding_data
        
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
            
        return dot_product / (magnitude_a * magnitude_b)

def load_test_cases(db_path: str, response_ids: List[str]) -> List[TestCase]:
    """Load test cases from the LLM logs database."""
    conn = sqlite3.connect(db_path)
    
    placeholders = ','.join('?' for _ in response_ids)
    query = f"""
    SELECT id, model, prompt, response, datetime_utc, 
           length(prompt) as prompt_len, length(response) as response_len
    FROM responses 
    WHERE id IN ({placeholders})
    ORDER BY datetime_utc
    """
    
    cursor = conn.execute(query, response_ids)
    test_cases = []
    
    for row in cursor:
        test_cases.append(TestCase(
            id=row[0],
            model=row[1], 
            prompt=row[2],
            response=row[3],
            datetime_utc=row[4],
            prompt_length=row[5],
            response_length=row[6]
        ))
    
    conn.close()
    return test_cases

async def run_compression_test(test_case: TestCase, strategy: str, 
                             compressor: PromptCompressor, 
                             embedder: EmbeddingComparer) -> CompressionResult:
    """Run a single compression/decompression test."""
    print(f"Testing {test_case.id} with strategy '{strategy}'...")
    
    # Step 1: Compress the prompt
    compressed = compressor.compress_prompt(test_case.prompt, strategy)
    
    # Step 2: Decompress back to expanded form
    decompressed = compressor.decompress_prompt(compressed, strategy)
    
    # Step 3: Get embeddings for original and decompressed (NOT compressed)
    original_embedding = embedder.get_embedding(test_case.prompt)
    decompressed_embedding = embedder.get_embedding(decompressed)
    
    # Step 4: Calculate similarity between original and decompressed
    similarity = embedder.cosine_similarity(original_embedding, decompressed_embedding)
    
    # Calculate compression metrics
    original_len = len(test_case.prompt)
    compressed_len = len(compressed)
    decompressed_len = len(decompressed)
    compression_ratio = compressed_len / original_len if original_len > 0 else 0
    decompression_ratio = decompressed_len / original_len if original_len > 0 else 0
    
    # Calculate token metrics
    original_tokens = TokenCounter.get_best_estimate(test_case.prompt)
    compressed_tokens = TokenCounter.get_best_estimate(compressed)
    decompressed_tokens = TokenCounter.get_best_estimate(decompressed)
    token_compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
    token_decompression_ratio = decompressed_tokens / original_tokens if original_tokens > 0 else 0
    
    return CompressionResult(
        original_prompt=test_case.prompt,
        compressed_prompt=compressed,
        decompressed_prompt=decompressed,
        compression_strategy=strategy,
        compression_model=compressor.compression_model,
        decompression_model=compressor.decompression_model,
        original_length=original_len,
        compressed_length=compressed_len,
        decompressed_length=decompressed_len,
        compression_ratio=compression_ratio,
        decompression_ratio=decompression_ratio,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        decompressed_tokens=decompressed_tokens,
        token_compression_ratio=token_compression_ratio,
        token_decompression_ratio=token_decompression_ratio,
        original_embedding=original_embedding,
        decompressed_embedding=decompressed_embedding,
        similarity_score=similarity
    )

def save_results(results: List[CompressionResult], output_file: str):
    """Save results to JSON file."""
    results_data = []
    for result in results:
        results_data.append({
            'compression_strategy': result.compression_strategy,
            'compression_model': result.compression_model,
            'decompression_model': result.decompression_model,
            'original_length': result.original_length,
            'compressed_length': result.compressed_length,
            'decompressed_length': result.decompressed_length,
            'compression_ratio': result.compression_ratio,
            'decompression_ratio': result.decompression_ratio,
            'original_tokens': result.original_tokens,
            'compressed_tokens': result.compressed_tokens,
            'decompressed_tokens': result.decompressed_tokens,
            'token_compression_ratio': result.token_compression_ratio,
            'token_decompression_ratio': result.token_decompression_ratio,
            'similarity_score': result.similarity_score,
            'original_prompt': result.original_prompt,
            'compressed_prompt': result.compressed_prompt,
            'decompressed_prompt': result.decompressed_prompt
        })
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Test prompt compression/decompression methods')
    parser.add_argument('--db-path', default='~/.config/io.datasette.llm/logs.db',
                      help='Path to LLM logs database')
    parser.add_argument('--response-ids', nargs='+', 
                      default=['01jz0cvyg7075fnqkkp2yb6jq7'],
                      help='Response IDs to test')
    parser.add_argument('--strategies', nargs='+',
                      default=['ultra_minimal', 'structured_compress', 'keyword_dense', 'telegram_style', 'code_like'],
                      help='Compression strategies to test')
    parser.add_argument('--output', default='improved_compression_results.json',
                      help='Output file for results')
    parser.add_argument('--compression-model', default='gemini-flash',
                      help='LLM model for compression')
    parser.add_argument('--decompression-model', default='gemini-flash',
                      help='LLM model for decompression')
    
    args = parser.parse_args()
    
    # Expand tilde in db path
    db_path = os.path.expanduser(args.db_path)
    
    print(f"Loading test cases from {db_path}...")
    test_cases = load_test_cases(db_path, args.response_ids)
    print(f"Loaded {len(test_cases)} test cases")
    
    compressor = PromptCompressor(args.compression_model, args.decompression_model)
    embedder = EmbeddingComparer()
    
    all_results = []
    
    # Run tests for each combination
    for test_case in test_cases:
        print(f"\nTesting case {test_case.id} (length: {test_case.prompt_length} chars, ~{TokenCounter.get_best_estimate(test_case.prompt)} tokens)")
        for strategy in args.strategies:
            try:
                result = asyncio.run(run_compression_test(
                    test_case, strategy, compressor, embedder))
                all_results.append(result)
                print(f"  {strategy}: {result.compression_ratio:.2f} compress ratio, "
                      f"{result.decompression_ratio:.2f} decompress ratio, "
                      f"{result.similarity_score:.3f} similarity")
            except Exception as e:
                print(f"  Error with {strategy}: {e}")
    
    # Save results
    save_results(all_results, args.output)
    print(f"\nResults saved to {args.output}")
    
    # Print summary
    print("\nSummary:")
    for strategy in args.strategies:
        strategy_results = [r for r in all_results if r.compression_strategy == strategy]
        if strategy_results:
            avg_compress_ratio = sum(r.compression_ratio for r in strategy_results) / len(strategy_results)
            avg_decompress_ratio = sum(r.decompression_ratio for r in strategy_results) / len(strategy_results)
            avg_similarity = sum(r.similarity_score for r in strategy_results) / len(strategy_results)
            print(f"  {strategy}: avg compress {avg_compress_ratio:.2f}, avg decompress {avg_decompress_ratio:.2f}, avg similarity {avg_similarity:.3f}")

if __name__ == '__main__':
    main()
