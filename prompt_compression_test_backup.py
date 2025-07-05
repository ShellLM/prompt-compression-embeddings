#!/usr/bin/env python3
"""
Prompt Compression Test (Backup Version)

This script tests prompt compression by:
1. Taking original prompts from the LLM logs database
2. Compressing them using various strategies
3. Comparing the compressed versions using embeddings
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

@dataclass
class CompressionResult:
    """Results from a compression test."""
    original_prompt: str
    compressed_prompt: str
    compression_ratio: float
    similarity_score: float
    strategy: str

class PromptCompressor:
    """Handles prompt compression using LLM."""
    
    def __init__(self, model: str = "gpt-4o-mini"):
        self.model = model
        
        # Define compression strategies
        self.strategies = {
            "minimal": "Compress this prompt to the absolute minimum words needed while preserving ALL key information:",
            "structured": "Compress this prompt into a structured, bullet-point format:",
            "keywords": "Extract the key concepts and compress into essential keywords:",
            "abbreviated": "Compress using abbreviations and shortened forms while keeping meaning:"
        }
    
    def compress_prompt(self, prompt: str, strategy: str) -> str:
        """Compress a prompt using the specified strategy."""
        if strategy not in self.strategies:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        compression_instruction = self.strategies[strategy]
        full_prompt = f"{compression_instruction}\n\n{prompt}"
        
        try:
            model_obj = llm.get_model(self.model)
            response = model_obj.prompt(full_prompt, temperature=0.1)
            return response.text().strip()
        except Exception as e:
            print(f"Error compressing with {self.model}: {e}")
            return prompt  # Return original if compression fails

class EmbeddingComparer:
    """Handles embedding generation and comparison."""
    
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

class CompressionTester:
    """Main tester class."""
    
    def __init__(self, db_path: str, model: str = "gpt-4o-mini", embedding_model: str = "text-embedding-3-small"):
        self.db_path = Path(db_path).expanduser()
        self.compressor = PromptCompressor(model)
        self.comparer = EmbeddingComparer(embedding_model)
    
    def get_test_prompts(self, min_length: int = 100, limit: int = 20) -> List[Tuple[str, str]]:
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
    
    def test_compression(self, prompt_id: str, prompt: str) -> List[CompressionResult]:
        """Test compression with all strategies for a single prompt."""
        results = []
        
        original_embedding = self.comparer.get_embedding(prompt)
        
        for strategy in self.compressor.strategies.keys():
            try:
                compressed = self.compressor.compress_prompt(prompt, strategy)
                compressed_embedding = self.comparer.get_embedding(compressed)
                
                compression_ratio = len(compressed) / len(prompt) if len(prompt) > 0 else 0
                similarity_score = self.comparer.cosine_similarity(original_embedding, compressed_embedding)
                
                result = CompressionResult(
                    original_prompt=prompt,
                    compressed_prompt=compressed,
                    compression_ratio=compression_ratio,
                    similarity_score=similarity_score,
                    strategy=strategy
                )
                
                results.append(result)
                print(f"  {strategy}: ratio={compression_ratio:.2f}, similarity={similarity_score:.3f}")
                
            except Exception as e:
                print(f"  Error with {strategy}: {e}")
        
        return results
    
    def run_tests(self, min_length: int = 100, limit: int = 20) -> List[CompressionResult]:
        """Run compression tests on multiple prompts."""
        print(f"Getting test prompts (min_length={min_length}, limit={limit})...")
        test_prompts = self.get_test_prompts(min_length, limit)
        print(f"Testing {len(test_prompts)} prompts...")
        
        all_results = []
        
        for i, (prompt_id, prompt) in enumerate(test_prompts):
            print(f"\nTesting prompt {i+1}/{len(test_prompts)}: {prompt_id}")
            results = self.test_compression(prompt_id, prompt)
            all_results.extend(results)
        
        return all_results

def main():
    parser = argparse.ArgumentParser(description='Test prompt compression (backup version)')
    parser.add_argument('--db', default='~/.config/io.datasette.llm/logs.db', help='Database path')
    parser.add_argument('--min-length', type=int, default=100, help='Minimum prompt length')
    parser.add_argument('--limit', type=int, default=20, help='Number of prompts to test')
    parser.add_argument('--output', default='compression_test_backup_results.json', help='Output file')
    parser.add_argument('--model', default='gpt-4o-mini', help='Compression model')
    parser.add_argument('--embedding-model', default='text-embedding-3-small', help='Embedding model')
    
    args = parser.parse_args()
    
    tester = CompressionTester(
        db_path=args.db,
        model=args.model,
        embedding_model=args.embedding_model
    )
    
    results = tester.run_tests(args.min_length, args.limit)
    
    # Convert to JSON-serializable format
    output_data = []
    for result in results:
        output_data.append({
            'original_prompt': result.original_prompt,
            'compressed_prompt': result.compressed_prompt,
            'compression_ratio': result.compression_ratio,
            'similarity_score': result.similarity_score,
            'strategy': result.strategy
        })
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nCompleted {len(results)} compression tests")
    print(f"Results saved to {args.output}")
    
    # Show summary statistics
    if results:
        by_strategy = {}
        for result in results:
            strategy = result.strategy
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(result)
        
        print("\nStrategy performance:")
        for strategy, strategy_results in by_strategy.items():
            avg_compression = sum(r.compression_ratio for r in strategy_results) / len(strategy_results)
            avg_similarity = sum(r.similarity_score for r in strategy_results) / len(strategy_results)
            print(f"  {strategy}: avg_compression={avg_compression:.2f}, avg_similarity={avg_similarity:.3f}")

if __name__ == '__main__':
    main()
