#!/usr/bin/env python3
"""
Prompt Compression Embeddings Test

This script tests different prompt compression methods by:
1. Loading test prompts from LLM logs database
2. Generating compressed versions using various strategies
3. Computing embeddings for original and compressed versions
4. Measuring semantic similarity between them
5. Evaluating compression effectiveness
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
    """Results from a compression attempt."""
    original_prompt: str
    compressed_prompt: str
    compression_strategy: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    original_embedding: List[float]
    compressed_embedding: List[float]
    similarity_score: float

class PromptCompressor:
    """Handles different prompt compression strategies using LLM."""
    
    def __init__(self, model: str = "claude-3.5-haiku"):
        self.model = model
        
    def compress_prompt(self, prompt: str, strategy: str) -> str:
        """Compress a prompt using the specified strategy."""
        strategies = {
            "summarize": "Summarize this prompt in 50% fewer words while preserving all key information and intent:",
            "extract_core": "Extract only the core request and essential context from this prompt:",
            "bullet_points": "Convert this prompt into concise bullet points covering all main points:",
            "minimal": "Rewrite this prompt using the minimum words necessary while keeping all meaning:",
            "keywords": "Extract the key terms and main request from this prompt, maintaining clarity:"
        }
        
        if strategy not in strategies:
            raise ValueError(f"Unknown compression strategy: {strategy}")
            
        compression_instruction = strategies[strategy]
        full_prompt = f"{compression_instruction}\n\n{prompt}"
        
        # Use llm CLI to get compressed version
        result = subprocess.run([
            'llm', '-m', self.model, 
            full_prompt
        ], capture_output=True, text=True, check=True)
        
        return result.stdout.strip()

class EmbeddingComparer:
    """Handles embedding generation and similarity comparison."""
    
    def __init__(self, embedding_model: str = "jina-v4"):
        self.embedding_model = embedding_model
        
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using llm CLI."""
        import json
        
        # Use shell to pipe text to llm embed
        cmd = f'echo {json.dumps(text)} | llm embed -m {self.embedding_model}'
        result = subprocess.run(['sh', '-c', cmd], capture_output=True, text=True, check=True)
        
        # Parse the JSON array output
        embedding_data = json.loads(result.stdout.strip())
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
    """Run a single compression test."""
    print(f"Testing {test_case.id} with strategy '{strategy}'...")
    
    # Compress the prompt
    compressed = compressor.compress_prompt(test_case.prompt, strategy)
    
    # Get embeddings for both versions
    original_embedding = embedder.get_embedding(test_case.prompt)
    compressed_embedding = embedder.get_embedding(compressed)
    
    # Calculate similarity
    similarity = embedder.cosine_similarity(original_embedding, compressed_embedding)
    
    # Calculate compression metrics
    original_len = len(test_case.prompt)
    compressed_len = len(compressed)
    compression_ratio = compressed_len / original_len if original_len > 0 else 0
    
    return CompressionResult(
        original_prompt=test_case.prompt,
        compressed_prompt=compressed,
        compression_strategy=strategy,
        original_length=original_len,
        compressed_length=compressed_len,
        compression_ratio=compression_ratio,
        original_embedding=original_embedding,
        compressed_embedding=compressed_embedding,
        similarity_score=similarity
    )

def save_results(results: List[CompressionResult], output_file: str):
    """Save results to JSON file."""
    results_data = []
    for result in results:
        results_data.append({
            'compression_strategy': result.compression_strategy,
            'original_length': result.original_length,
            'compressed_length': result.compressed_length,
            'compression_ratio': result.compression_ratio,
            'similarity_score': result.similarity_score,
            'original_prompt': result.original_prompt,
            'compressed_prompt': result.compressed_prompt
        })
    
    with open(output_file, 'w') as f:
        json.dump(results_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description='Test prompt compression methods')
    parser.add_argument('--db-path', default='~/.config/io.datasette.llm/logs.db',
                      help='Path to LLM logs database')
    parser.add_argument('--response-ids', nargs='+', 
                      default=['01jz0cvyg7075fnqkkp2yb6jq7', '01jz09kjeqkg2ktdqedyych20e', 
                              '01jz0jwxpe8tbytrmey4b15gxa', '01jz3a906ksjkh0fwnbx4c9wvn'],
                      help='Response IDs to test')
    parser.add_argument('--strategies', nargs='+',
                      default=['summarize', 'extract_core', 'bullet_points', 'minimal', 'keywords'],
                      help='Compression strategies to test')
    parser.add_argument('--output', default='compression_results.json',
                      help='Output file for results')
    parser.add_argument('--model', default='claude-3.5-haiku',
                      help='LLM model for compression')
    
    args = parser.parse_args()
    
    # Expand tilde in db path
    db_path = os.path.expanduser(args.db_path)
    
    print(f"Loading test cases from {db_path}...")
    test_cases = load_test_cases(db_path, args.response_ids)
    print(f"Loaded {len(test_cases)} test cases")
    
    compressor = PromptCompressor(args.model)
    embedder = EmbeddingComparer()
    
    all_results = []
    
    # Run tests for each combination
    for test_case in test_cases:
        print(f"\nTesting case {test_case.id} (length: {test_case.prompt_length})")
        for strategy in args.strategies:
            try:
                result = asyncio.run(run_compression_test(
                    test_case, strategy, compressor, embedder))
                all_results.append(result)
                print(f"  {strategy}: {result.compression_ratio:.2f} ratio, "
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
            avg_ratio = sum(r.compression_ratio for r in strategy_results) / len(strategy_results)
            avg_similarity = sum(r.similarity_score for r in strategy_results) / len(strategy_results)
            print(f"  {strategy}: avg ratio {avg_ratio:.2f}, avg similarity {avg_similarity:.3f}")

if __name__ == '__main__':
    main()
