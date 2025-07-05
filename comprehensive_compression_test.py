#!/usr/bin/env python3
"""
Comprehensive Prompt Compression Test Suite

This script runs compression tests using multiple models and strategies in batches.
It loads exemplars from the database exploration and tests compression/decompression cycles.
"""

import json
import subprocess
import argparse
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, asdict
import random
import tempfile

@dataclass
class TestConfiguration:
    """Configuration for a compression test."""
    compression_model: str
    decompression_model: str
    strategy: str
    batch_size: int = 5

@dataclass
class BatchResult:
    """Results from a batch of compression tests."""
    config: TestConfiguration
    results: List[Dict]
    batch_stats: Dict

class ComprehensiveTestRunner:
    """Runs comprehensive compression tests with multiple models and strategies."""
    
    def __init__(self, exemplars_file: str):
        self.exemplars_file = exemplars_file
        self.load_exemplars()
        
    def load_exemplars(self):
        """Load exemplars from JSON file."""
        with open(self.exemplars_file, 'r') as f:
            self.exemplars = json.load(f)
        print(f"Loaded {len(self.exemplars)} exemplars")
    
    def get_test_configurations(self) -> List[TestConfiguration]:
        """Get all test configurations to run."""
        # Multiple model combinations
        model_pairs = [
            ("gemini-flash", "gemini-flash"),
            ("gemini-flash-lite", "gemini-flash-lite"),
            ("gemini-flash", "claude-4-sonnet"),
            ("claude-4-sonnet", "gemini-flash"),
            ("gemini-pro", "gemini-pro"),
        ]
        
        # Multiple compression strategies
        strategies = [
            "ultra_minimal",
            "structured_compress", 
            "keyword_dense",
            "telegram_style",
            "code_like"
        ]
        
        configurations = []
        for compression_model, decompression_model in model_pairs:
            for strategy in strategies:
                configurations.append(TestConfiguration(
                    compression_model=compression_model,
                    decompression_model=decompression_model,
                    strategy=strategy,
                    batch_size=5
                ))
        
        return configurations
    
    def run_single_test(self, exemplar: Dict, config: TestConfiguration) -> Optional[Dict]:
        """Run a single compression test."""
        try:
            # Step 1: Compress
            compression_result = self.compress_text(exemplar['text'], config.strategy, config.compression_model)
            if not compression_result:
                return None
            
            # Step 2: Decompress
            decompression_result = self.decompress_text(compression_result, config.strategy, config.decompression_model)
            if not decompression_result:
                return None
            
            # Step 3: Get embeddings and similarity
            original_embedding = self.get_embedding(exemplar['text'])
            decompressed_embedding = self.get_embedding(decompression_result)
            
            if not original_embedding or not decompressed_embedding:
                return None
            
            similarity = self.cosine_similarity(original_embedding, decompressed_embedding)
            
            # Calculate metrics
            original_len = len(exemplar['text'])
            compressed_len = len(compression_result)
            decompressed_len = len(decompression_result)
            
            return {
                'exemplar_id': exemplar['id'],
                'exemplar_category': exemplar['assessment']['category'],
                'exemplar_density': exemplar['density_score'],
                'original_length': original_len,
                'compressed_length': compressed_len,
                'decompressed_length': decompressed_len,
                'compression_ratio': compressed_len / original_len if original_len > 0 else 0,
                'decompression_ratio': decompressed_len / original_len if original_len > 0 else 0,
                'similarity_score': similarity,
                'original_text': exemplar['text'],
                'compressed_text': compression_result,
                'decompressed_text': decompression_result,
                'compression_model': config.compression_model,
                'decompression_model': config.decompression_model,
                'strategy': config.strategy
            }
            
        except Exception as e:
            print(f"Error in single test: {e}")
            return None
    
    def compress_text(self, text: str, strategy: str, model: str) -> Optional[str]:
        """Compress text using specified strategy and model."""
        strategies = {
            "ultra_minimal": "Compress this text to the absolute minimum words needed while preserving ALL key information. Use abbreviations, remove redundant words, keep only essential context:",
            "structured_compress": "Compress this text into a structured format (bullets, key-value pairs, etc.) that captures all information in minimal space:",
            "keyword_dense": "Extract and compress this text into dense keyword clusters that preserve all meaning and relationships:",
            "telegram_style": "Compress this text like a telegram - every word costs money, but preserve all essential information:",
            "code_like": "Compress this text into a code-like structure (pseudo-code, structured notation) that preserves all logic and information:"
        }
        
        instruction = strategies[strategy]
        full_prompt = f"{instruction}\n\n{text}"
        
        try:
            result = subprocess.run([
                'llm', '-m', model, full_prompt
            ], capture_output=True, text=True, check=True, timeout=180)
            return result.stdout.strip()
        except Exception as e:
            print(f"Compression failed with {model}: {e}")
            return None
    
    def decompress_text(self, compressed_text: str, strategy: str, model: str) -> Optional[str]:
        """Decompress text back to expanded form."""
        decompression_instruction = f"""
        This is a compressed text that was created using the '{strategy}' compression strategy.
        Please expand it back to a full, natural, detailed text that captures all the original information and intent.
        Make it complete and well-structured, filling in natural language while preserving all the compressed information.
        
        Compressed text: {compressed_text}
        """
        
        try:
            result = subprocess.run([
                'llm', '-m', model, decompression_instruction
            ], capture_output=True, text=True, check=True, timeout=180)
            return result.stdout.strip()
        except Exception as e:
            print(f"Decompression failed with {model}: {e}")
            return None
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for text."""
        try:
            # Use a temporary file to avoid shell argument length limits
            with tempfile.NamedTemporaryFile(mode='w', delete=True, encoding='utf-8') as temp_f:
                temp_f.write(text)
                temp_f.flush() # Ensure data is written to disk before the subprocess reads it
                
                cmd = f'cat {temp_f.name} | llm embed -m jina-v4 2>/dev/null'
                result = subprocess.run(['sh', '-c', cmd], capture_output=True, text=True, check=True, timeout=180)
            
            output_lines = result.stdout.strip().split('\n')
            json_line = output_lines[-1]
            embedding_data = json.loads(json_line)
            return embedding_data
        except Exception as e:
            print(f"Embedding failed: {e}")
            return None
    
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math
        
        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        magnitude_a = math.sqrt(sum(a * a for a in vec_a))
        magnitude_b = math.sqrt(sum(b * b for b in vec_b))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
            
        return dot_product / (magnitude_a * magnitude_b)
    
    def run_batch(self, config: TestConfiguration, exemplars: List[Dict]) -> BatchResult:
        """Run a batch of tests with the given configuration."""
        print(f"\nRunning batch: {config.compression_model} -> {config.decompression_model}, strategy: {config.strategy}")
        
        batch_results = []
        for i, exemplar in enumerate(exemplars):
            print(f"  Test {i+1}/{len(exemplars)}: {exemplar['id']}")
            result = self.run_single_test(exemplar, config)
            if result:
                batch_results.append(result)
                print(f"    ✓ Compression: {result['compression_ratio']:.2f}, Similarity: {result['similarity_score']:.3f}")
            else:
                print(f"    ✗ Failed")
        
        # Calculate batch statistics
        if batch_results:
            batch_stats = {
                'total_tests': len(exemplars),
                'successful_tests': len(batch_results),
                'success_rate': len(batch_results) / len(exemplars),
                'avg_compression_ratio': sum(r['compression_ratio'] for r in batch_results) / len(batch_results),
                'avg_decompression_ratio': sum(r['decompression_ratio'] for r in batch_results) / len(batch_results),
                'avg_similarity': sum(r['similarity_score'] for r in batch_results) / len(batch_results),
                'min_similarity': min(r['similarity_score'] for r in batch_results),
                'max_similarity': max(r['similarity_score'] for r in batch_results),
            }
        else:
            batch_stats = {
                'total_tests': len(exemplars),
                'successful_tests': 0,
                'success_rate': 0.0,
                'avg_compression_ratio': 0.0,
                'avg_decompression_ratio': 0.0,
                'avg_similarity': 0.0,
                'min_similarity': 0.0,
                'max_similarity': 0.0,
            }
        
        return BatchResult(config=config, results=batch_results, batch_stats=batch_stats)
    
    def run_comprehensive_test(self, output_file: str = "comprehensive_results.json"):
        """Run comprehensive test across all configurations."""
        configurations = self.get_test_configurations()
        print(f"Running {len(configurations)} different configurations")
        
        # Shuffle exemplars and select batches
        random.shuffle(self.exemplars)
        
        all_batch_results = []
        
        for i, config in enumerate(configurations):
            print(f"\n=== Configuration {i+1}/{len(configurations)} ===")
            
            # Select batch of exemplars
            start_idx = (i * config.batch_size) % len(self.exemplars)
            end_idx = min(start_idx + config.batch_size, len(self.exemplars))
            batch_exemplars = self.exemplars[start_idx:end_idx]
            
            # If we don't have enough exemplars, wrap around
            if len(batch_exemplars) < config.batch_size:
                remaining = config.batch_size - len(batch_exemplars)
                batch_exemplars.extend(self.exemplars[:remaining])
            
            batch_result = self.run_batch(config, batch_exemplars)
            all_batch_results.append(batch_result)
            
            # Save intermediate results
            self.save_results(all_batch_results, output_file)
            
            # Small delay to avoid overwhelming the APIs
            time.sleep(2)
        
        # Final save and summary
        self.save_results(all_batch_results, output_file)
        self.print_summary(all_batch_results)
    
    def save_results(self, batch_results: List[BatchResult], output_file: str):
        """Save all batch results to JSON file."""
        output_data = {
            'timestamp': time.time(),
            'total_batches': len(batch_results),
            'batch_results': []
        }
        
        for batch_result in batch_results:
            output_data['batch_results'].append({
                'config': asdict(batch_result.config),
                'batch_stats': batch_result.batch_stats,
                'results': batch_result.results
            })
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {output_file}")
    
    def print_summary(self, batch_results: List[BatchResult]):
        """Print summary of all results."""
        print("\n" + "="*60)
        print("COMPREHENSIVE TEST SUMMARY")
        print("="*60)
        
        # Group by strategy
        by_strategy = {}
        for batch_result in batch_results:
            strategy = batch_result.config.strategy
            if strategy not in by_strategy:
                by_strategy[strategy] = []
            by_strategy[strategy].append(batch_result)
        
        for strategy, batches in by_strategy.items():
            print(f"\n{strategy.upper()}:")
            for batch in batches:
                config = batch.config
                stats = batch.batch_stats
                print(f"  {config.compression_model} -> {config.decompression_model}: "
                      f"success={stats['success_rate']:.2f}, "
                      f"compress={stats['avg_compression_ratio']:.2f}, "
                      f"similarity={stats['avg_similarity']:.3f}")

def main():
    parser = argparse.ArgumentParser(description='Run comprehensive compression tests')
    parser.add_argument('--exemplars', default='high_quality_exemplars.json',
                      help='JSON file with exemplars')
    parser.add_argument('--output', default='comprehensive_results.json',
                      help='Output file for results')
    
    args = parser.parse_args()
    
    runner = ComprehensiveTestRunner(args.exemplars)
    runner.run_comprehensive_test(args.output)

if __name__ == '__main__':
    main()
