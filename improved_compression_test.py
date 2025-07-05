#!/usr/bin/env python3
"""
Improved Prompt Compression Test

This script tests different compression strategies by:
1. Compressing prompts using various strategies
2. Decompressing them back to full form
3. Comparing semantic similarity using embeddings
4. Evaluating compression ratios and quality
"""

import os
import sqlite3
import json
import llm
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path
import time

@dataclass
class CompressionResult:
    """Results from a compression test."""
    original_prompt: str
    compressed_prompt: str
    decompressed_prompt: str
    compression_ratio: float
    similarity_score: float
    strategy: str
    compression_model: str
    decompression_model: str
    compression_time: float
    decompression_time: float

class PromptCompressor:
    """Handles prompt compression using various strategies."""
    
    def __init__(self, compression_model: str = "gpt-4o-mini", decompression_model: str = "gpt-4o-mini"):
        self.compression_model = compression_model
        self.decompression_model = decompression_model
        
        # Define compression strategies
        self.strategies = {
            "minimal": "Compress this prompt to the absolute minimum words needed while preserving ALL key information. Use abbreviations, remove redundant words, keep only essential context:",
            "structured": "Compress this prompt into a structured format (bullets, key-value pairs, etc.) that captures all information in minimal space:",
            "keywords": "Extract and compress this prompt into dense keyword clusters that preserve all meaning and relationships:",
            "telegram": "Compress this prompt like a telegram - every word costs money, but preserve all essential information:",
            "json": "Compress this prompt into a JSON structure that preserves all logic and information:"
        }
    
    def compress_prompt(self, prompt: str, strategy: str) -> str:
        """Compress a prompt using the specified strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        compression_instruction = self.strategies[strategy]
        full_prompt = f"{compression_instruction}\n\n{prompt}"
        
        try:
            model_obj = llm.get_model(self.compression_model)
            response = model_obj.prompt(full_prompt, temperature=0.1)
            return response.text().strip()
        except Exception as e:
            print(f"Error compressing with {self.compression_model}: {e}")
            return prompt  # Return original if compression fails
    
    def decompress_prompt(self, compressed_prompt: str, strategy: str) -> str:
        """Decompress a compressed prompt back to full form."""
        decompression_instruction = f"""
        This is a compressed prompt that was created using the '{strategy}' compression strategy.
        Please expand it back to a full, natural, detailed prompt that captures all the original information and intent.
        Make it complete and well-structured, filling in natural language while preserving all the compressed information.
        
        Compressed prompt: {compressed_prompt}
        """
        
        try:
            model_obj = llm.get_model(self.decompression_model)
            response = model_obj.prompt(decompression_instruction, temperature=0.1)
            return response.text().strip()
        except Exception as e:
            print(f"Error decompressing with {self.decompression_model}: {e}")
            return compressed_prompt  # Return compressed if decompression fails

class EmbeddingComparer:
    """Handles embedding generation and similarity comparison."""
    
    def __init__(self, embedding_model: str = "text-embedding-3-small"):
        self.embedding_model = embedding_model
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using llm Python API."""
        try:
            embedding_model_obj = llm.get_embedding_model(self.embedding_model)
            embedding_data = embedding_model_obj.embed(text)
            return embedding_data
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return []
    
    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2:
            return 0.0
        
        # Convert to numpy arrays
        a = np.array(vec1)
        b = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def compare_prompts(self, original: str, decompressed: str) -> float:
        """Compare two prompts using embedding similarity."""
        original_embedding = self.get_embedding(original)
        decompressed_embedding = self.get_embedding(decompressed)
        
        return self.cosine_similarity(original_embedding, decompressed_embedding)

class CompressionTester:
    """Main class for running compression tests."""
    
    def __init__(self, db_path: str, compression_model: str = "gpt-4o-mini", 
                 decompression_model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        self.db_path = Path(db_path).expanduser()
        self.compressor = PromptCompressor(compression_model, decompression_model)
        self.comparer = EmbeddingComparer(embedding_model)
    
    def get_test_prompts(self, min_length: int = 200, limit: int = 10) -> List[Tuple[str, str]]:
        """Get test prompts from the database."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT id, prompt FROM responses 
            WHERE LENGTH(prompt) > ? 
            ORDER BY RANDOM() 
            LIMIT ?
            """
            cursor = conn.execute(query, (min_length, limit))
            return cursor.fetchall()
    
    def run_compression_test(self, prompt_id: str, prompt: str, strategy: str) -> CompressionResult:
        """Run a single compression test."""
        print(f"Testing prompt {prompt_id} with strategy '{strategy}'...")
        
        # Compress
        start_time = time.time()
        compressed = self.compressor.compress_prompt(prompt, strategy)
        compression_time = time.time() - start_time
        
        # Decompress
        start_time = time.time()
        decompressed = self.compressor.decompress_prompt(compressed, strategy)
        decompression_time = time.time() - start_time
        
        # Calculate metrics
        compression_ratio = len(compressed) / len(prompt) if len(prompt) > 0 else 0
        similarity_score = self.comparer.compare_prompts(prompt, decompressed)
        
        return CompressionResult(
            original_prompt=prompt,
            compressed_prompt=compressed,
            decompressed_prompt=decompressed,
            compression_ratio=compression_ratio,
            similarity_score=similarity_score,
            strategy=strategy,
            compression_model=self.compressor.compression_model,
            decompression_model=self.compressor.decompression_model,
            compression_time=compression_time,
            decompression_time=decompression_time
        )
    
    def run_comprehensive_test(self, min_length: int = 200, limit: int = 10) -> List[CompressionResult]:
        """Run comprehensive tests across all strategies."""
        print(f"Running comprehensive compression tests...")
        
        # Get test prompts
        test_prompts = self.get_test_prompts(min_length, limit)
        print(f"Testing {len(test_prompts)} prompts with {len(self.compressor.strategies)} strategies")
        
        results = []
        
        for prompt_id, prompt in test_prompts:
            for strategy in self.compressor.strategies.keys():
                try:
                    result = self.run_compression_test(prompt_id, prompt, strategy)
                    results.append(result)
                    print(f"  ✓ {strategy}: compression={result.compression_ratio:.2f}, similarity={result.similarity_score:.3f}")
                except Exception as e:
                    print(f"  ✗ {strategy}: {e}")
        
        return results

def main():
    parser = argparse.ArgumentParser(description='Test prompt compression strategies')
    parser.add_argument('--db', default='~/.config/io.datasette.llm/logs.db', help='Database path')
    parser.add_argument('--min-length', type=int, default=200, help='Minimum prompt length')
    parser.add_argument('--limit', type=int, default=10, help='Number of prompts to test')
    parser.add_argument('--output', default='test_results.json', help='Output file')
    parser.add_argument('--compression-model', default='gpt-4o-mini', help='Compression model')
    parser.add_argument('--decompression-model', default='gpt-4o-mini', help='Decompression model')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='Embedding model')
    
    args = parser.parse_args()
    
    tester = CompressionTester(
        db_path=args.db,
        compression_model=args.compression_model,
        decompression_model=args.decompression_model,
        embedding_model=args.embedding_model
    )
    
    results = tester.run_comprehensive_test(args.min_length, args.limit)
    
    # Convert to JSON-serializable format
    output_data = []
    for result in results:
        output_data.append({
            'original_prompt': result.original_prompt,
            'compressed_prompt': result.compressed_prompt,
            'decompressed_prompt': result.decompressed_prompt,
            'compression_ratio': result.compression_ratio,
            'similarity_score': result.similarity_score,
            'strategy': result.strategy,
            'compression_model': result.compression_model,
            'decompression_model': result.decompression_model,
            'compression_time': result.compression_time,
            'decompression_time': result.decompression_time
        })
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCompleted {len(results)} compression tests")
    print(f"Results saved to {args.output}")
    
    # Show summary
    if results:
        # Group by strategy
        by_strategy = {}
        for result in results:
            strategy = result.strategy
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(result)
        
        print("\nStrategy performance summary:")
        for strategy, strategy_results in by_strategy.items():
            avg_compression = sum(r.compression_ratio for r in strategy_results) / len(strategy_results)
            avg_similarity = sum(r.similarity_score for r in strategy_results) / len(strategy_results)
            print(f"  {strategy}: avg_compression={avg_compression:.2f}, avg_similarity={avg_similarity:.3f}")

if __name__ == '__main__':
    main()
