#!/usr/bin/env python3
"""
Generate Compression Prompts using Multiple Models

This script uses different models to generate and refine compression prompts
for better compression strategies.
"""

import subprocess
import json
import argparse
from typing import List, Dict
from dataclasses import dataclass

@dataclass
class PromptCandidate:
    """A compression prompt candidate."""
    strategy_name: str
    prompt_text: str
    generator_model: str
    refinement_model: str
    effectiveness_score: float

class PromptGenerator:
    """Generate compression prompts using multiple models."""
    
    def __init__(self):
        self.generator_models = ["gemini-flash", "gemini-flash-lite", "claude-4-sonnet", "gemini-pro"]
        self.refinement_models = ["claude-4-sonnet", "gemini-pro"]
    
    def generate_base_prompts(self, strategy_description: str) -> List[str]:
        """Generate base compression prompts using multiple models."""
        base_prompt = f"""
        Create a compression prompt for the following strategy: {strategy_description}
        
        The prompt should instruct an AI to compress text while preserving all essential information.
        The prompt should be clear, specific, and effective.
        
        Return only the compression prompt itself, no additional explanation.
        """
        
        prompts = []
        for model in self.generator_models:
            try:
                result = subprocess.run([
                    'llm', '-m', model, base_prompt
                ], capture_output=True, text=True, check=True, timeout=30)
                
                prompt = result.stdout.strip()
                if prompt:
                    prompts.append(prompt)
                    print(f"✓ Generated prompt with {model}")
                else:
                    print(f"✗ Empty prompt from {model}")
            except Exception as e:
                print(f"✗ Error with {model}: {e}")
        
        return prompts
    
    def refine_prompt(self, base_prompt: str, refinement_model: str) -> str:
        """Refine a compression prompt using a refinement model."""
        refinement_instruction = f"""
        Improve this compression prompt to make it more effective at preserving information while maximizing compression.
        
        Make it more specific, clear, and actionable. Consider:
        - Explicitly mentioning to preserve ALL key information
        - Providing specific techniques (abbreviations, structured format, etc.)
        - Being clear about what can and cannot be removed
        - Ensuring the compressed output remains comprehensible
        
        Original prompt: {base_prompt}
        
        Return only the improved prompt, no additional explanation.
        """
        
        try:
            result = subprocess.run([
                'llm', '-m', refinement_model, refinement_instruction
            ], capture_output=True, text=True, check=True, timeout=30)
            
            refined = result.stdout.strip()
            return refined if refined else base_prompt
        except Exception as e:
            print(f"Refinement failed with {refinement_model}: {e}")
            return base_prompt
    
    def evaluate_prompt_effectiveness(self, prompt: str, test_text: str) -> float:
        """Evaluate how effective a compression prompt is using a quick test."""
        try:
            # Test compression
            compression_result = subprocess.run([
                'llm', '-m', 'gemini-flash-lite', f"{prompt}\n\n{test_text}"
            ], capture_output=True, text=True, check=True, timeout=30)
            
            compressed = compression_result.stdout.strip()
            
            if not compressed:
                return 0.0
            
            # Calculate basic metrics
            original_len = len(test_text)
            compressed_len = len(compressed)
            compression_ratio = compressed_len / original_len if original_len > 0 else 0
            
            # Score based on compression ratio (lower is better, but not too low)
            if compression_ratio > 0.8:  # Too little compression
                return 0.3
            elif compression_ratio < 0.2:  # Too much compression, likely lost info
                return 0.5
            else:  # Good compression range
                return 1.0 - abs(compression_ratio - 0.5)  # Optimal around 50%
            
        except Exception as e:
            print(f"Evaluation failed: {e}")
            return 0.0
    
    def generate_strategy_prompts(self, strategy_name: str, strategy_description: str, 
                                test_texts: List[str]) -> List[PromptCandidate]:
        """Generate and refine prompts for a specific strategy."""
        print(f"\nGenerating prompts for strategy: {strategy_name}")
        
        # Generate base prompts
        base_prompts = self.generate_base_prompts(strategy_description)
        
        candidates = []
        
        for base_prompt in base_prompts:
            for refinement_model in self.refinement_models:
                # Refine the prompt
                refined_prompt = self.refine_prompt(base_prompt, refinement_model)
                
                # Test effectiveness on sample texts
                total_score = 0.0
                for test_text in test_texts:
                    score = self.evaluate_prompt_effectiveness(refined_prompt, test_text)
                    total_score += score
                
                avg_score = total_score / len(test_texts) if test_texts else 0.0
                
                candidate = PromptCandidate(
                    strategy_name=strategy_name,
                    prompt_text=refined_prompt,
                    generator_model="multiple",
                    refinement_model=refinement_model,
                    effectiveness_score=avg_score
                )
                
                candidates.append(candidate)
                print(f"  ✓ Generated candidate (score: {avg_score:.2f})")
        
        # Sort by effectiveness
        candidates.sort(key=lambda x: x.effectiveness_score, reverse=True)
        
        return candidates
    
    def generate_all_prompts(self, test_texts: List[str]) -> Dict[str, List[PromptCandidate]]:
        """Generate prompts for all compression strategies."""
        strategies = {
            "ultra_minimal": "Create the most minimal version possible while preserving all key information",
            "structured_compress": "Convert to structured format (bullets, key-value pairs) for maximum density",
            "keyword_dense": "Extract into dense keyword clusters that preserve meaning and relationships",
            "telegram_style": "Compress like a telegram where every word costs money",
            "code_like": "Transform into code-like structured notation that preserves all logic",
            "hierarchical": "Create a hierarchical structure that captures all information levels",
            "symbolic": "Use symbols and abbreviations to maximize compression",
            "contextual": "Preserve context and relationships while minimizing redundancy"
        }
        
        all_candidates = {}
        
        for strategy_name, strategy_description in strategies.items():
            candidates = self.generate_strategy_prompts(strategy_name, strategy_description, test_texts)
            all_candidates[strategy_name] = candidates
        
        return all_candidates
    
    def save_prompts(self, candidates_dict: Dict[str, List[PromptCandidate]], output_file: str):
        """Save generated prompts to JSON file."""
        output_data = {
            'timestamp': time.time(),
            'strategies': {}
        }
        
        for strategy_name, candidates in candidates_dict.items():
            output_data['strategies'][strategy_name] = [
                {
                    'prompt_text': c.prompt_text,
                    'generator_model': c.generator_model,
                    'refinement_model': c.refinement_model,
                    'effectiveness_score': c.effectiveness_score
                }
                for c in candidates
            ]
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\nSaved prompts to {output_file}")
    
    def print_best_prompts(self, candidates_dict: Dict[str, List[PromptCandidate]]):
        """Print the best prompt for each strategy."""
        print("\n" + "="*60)
        print("BEST COMPRESSION PROMPTS")
        print("="*60)
        
        for strategy_name, candidates in candidates_dict.items():
            if candidates:
                best = candidates[0]  # Already sorted by effectiveness
                print(f"\n{strategy_name.upper()} (score: {best.effectiveness_score:.2f}):")
                print(f"Refined by: {best.refinement_model}")
                print(f"Prompt: {best.prompt_text}")
                print("-" * 40)

def main():
    parser = argparse.ArgumentParser(description='Generate compression prompts using multiple models')
    parser.add_argument('--test-texts-file', default='exemplars.json',
                      help='JSON file with test texts')
    parser.add_argument('--output', default='generated_prompts.json',
                      help='Output file for generated prompts')
    parser.add_argument('--sample-size', type=int, default=3,
                      help='Number of test texts to use for evaluation')
    
    args = parser.parse_args()
    
    # Load test texts
    test_texts = []
    try:
        with open(args.test_texts_file, 'r') as f:
            data = json.load(f)
            # Extract prompts from exemplars
            for item in data[:args.sample_size]:
                if 'prompt' in item and item['prompt']:
                    test_texts.append(item['prompt'])
    except Exception as e:
        print(f"Error loading test texts: {e}")
        # Use some default test texts
        test_texts = [
            "Please analyze this complex technical document and provide a comprehensive summary of the key findings and recommendations.",
            "I need help debugging this Python script that's throwing an error when trying to process multiple file formats.",
            "Create a detailed project plan for implementing a new machine learning pipeline with proper testing and deployment strategies."
        ]
    
    print(f"Using {len(test_texts)} test texts for evaluation")
    
    generator = PromptGenerator()
    candidates_dict = generator.generate_all_prompts(test_texts)
    
    generator.save_prompts(candidates_dict, args.output)
    generator.print_best_prompts(candidates_dict)

if __name__ == '__main__':
    import time
    main()
