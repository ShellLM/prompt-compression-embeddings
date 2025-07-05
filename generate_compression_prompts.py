#!/usr/bin/env python3
"""
Generate Compression Prompts using Multiple Models

This script uses different models to generate and refine compression prompts
for better compression strategies.
"""

import llm
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
        self.generator_models = ["claude-3.7-sonnet", "gpt-4o", "gpt-4o-mini"]
        self.refinement_models = ["claude-3.7-sonnet", "gpt-4o"]
    
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
                model_obj = llm.get_model(model)
                response = model_obj.prompt(base_prompt.strip(), temperature=0.7)
                prompt = response.text().strip()
                if prompt:
                    prompts.append(prompt)
                    print(f"✓ Generated prompt using {model}")
            except Exception as e:
                print(f"✗ Failed to generate prompt with {model}: {e}")
        
        return prompts
    
    def refine_prompt(self, base_prompt: str, strategy_name: str, refinement_model: str) -> str:
        """Refine a base prompt using a refinement model."""
        refinement_instruction = f"""
        Refine the following compression prompt for the "{strategy_name}" strategy:
        
        Original prompt: {base_prompt}
        
        Make it more specific, clear, and effective while keeping the same core strategy.
        Focus on clarity and actionable instructions.
        
        Return only the refined prompt, no additional explanation.
        """
        
        try:
            model_obj = llm.get_model(refinement_model)
            response = model_obj.prompt(refinement_instruction.strip(), temperature=0.3)
            refined = response.text().strip()
            return refined if refined else base_prompt
        except Exception as e:
            print(f"✗ Failed to refine prompt with {refinement_model}: {e}")
            return base_prompt
    
    def evaluate_prompt_effectiveness(self, prompt: str, test_text: str) -> float:
        """Evaluate how effective a compression prompt is."""
        try:
            # Test compression
            model_obj = llm.get_model("gpt-4o-mini")
            full_prompt = f"{prompt}\n\n{test_text}"
            response = model_obj.prompt(full_prompt, temperature=0.1)
            compressed = response.text().strip()
            
            if not compressed:
                return 0.0
            
            # Simple effectiveness metric: compression ratio
            original_len = len(test_text)
            compressed_len = len(compressed)
            compression_ratio = compressed_len / original_len if original_len > 0 else 0
            
            # Effectiveness score: prefer higher compression (lower ratio)
            # Score between 0 and 1, higher is better
            effectiveness = max(0, 1 - compression_ratio)
            
            return effectiveness
            
        except Exception as e:
            print(f"✗ Failed to evaluate prompt effectiveness: {e}")
            return 0.0
    
    def generate_prompt_candidates(self, strategies: Dict[str, str], test_text: str) -> List[PromptCandidate]:
        """Generate and evaluate compression prompt candidates."""
        candidates = []
        
        for strategy_name, strategy_description in strategies.items():
            print(f"\nProcessing strategy: {strategy_name}")
            
            # Generate base prompts
            base_prompts = self.generate_base_prompts(strategy_description)
            
            # Refine and evaluate each prompt
            for i, base_prompt in enumerate(base_prompts):
                for refinement_model in self.refinement_models:
                    try:
                        refined_prompt = self.refine_prompt(base_prompt, strategy_name, refinement_model)
                        effectiveness = self.evaluate_prompt_effectiveness(refined_prompt, test_text)
                        
                        candidate = PromptCandidate(
                            strategy_name=strategy_name,
                            prompt_text=refined_prompt,
                            generator_model=self.generator_models[i % len(self.generator_models)],
                            refinement_model=refinement_model,
                            effectiveness_score=effectiveness
                        )
                        
                        candidates.append(candidate)
                        print(f"  ✓ Candidate: {refinement_model} (effectiveness: {effectiveness:.3f})")
                        
                    except Exception as e:
                        print(f"  ✗ Failed to process candidate: {e}")
        
        return candidates

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Generate compression prompts')
    parser.add_argument('--output', default='generated_prompts.json', help='Output file')
    parser.add_argument('--test-text', default='', help='Test text for evaluation')
    
    args = parser.parse_args()
    
    # Default test text if none provided
    if not args.test_text:
        args.test_text = """
        Machine learning is a subset of artificial intelligence that focuses on the development of algorithms
        and statistical models that enable computers to learn and make decisions from data without being
        explicitly programmed for every task. The field encompasses various techniques including supervised
        learning, unsupervised learning, and reinforcement learning, each with distinct approaches to
        pattern recognition and prediction.
        """
    
    # Define compression strategies
    strategies = {
        "keyword_extraction": "Extract only the most important keywords and key phrases while maintaining meaning",
        "bullet_point_summary": "Convert text to concise bullet points capturing essential information",
        "acronym_dense": "Use acronyms and abbreviations to compress text while preserving all information",
        "structural_compression": "Compress using structured formats like JSON or key-value pairs",
        "telegram_style": "Compress as if every word costs money, keeping only essential information"
    }
    
    generator = PromptGenerator()
    candidates = generator.generate_prompt_candidates(strategies, args.test_text)
    
    # Sort by effectiveness
    candidates.sort(key=lambda x: x.effectiveness_score, reverse=True)
    
    # Convert to dict for JSON serialization
    results = {
        'generation_info': {
            'total_candidates': len(candidates),
            'strategies': list(strategies.keys()),
            'generator_models': generator.generator_models,
            'refinement_models': generator.refinement_models
        },
        'candidates': [
            {
                'strategy_name': c.strategy_name,
                'prompt_text': c.prompt_text,
                'generator_model': c.generator_model,
                'refinement_model': c.refinement_model,
                'effectiveness_score': c.effectiveness_score
            }
            for c in candidates
        ]
    }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nGenerated {len(candidates)} prompt candidates")
    print(f"Results saved to {args.output}")
    
    # Show top candidates
    print("\nTop 3 candidates:")
    for i, candidate in enumerate(candidates[:3]):
        print(f"{i+1}. {candidate.strategy_name} ({candidate.effectiveness_score:.3f})")
        print(f"   Generator: {candidate.generator_model}, Refiner: {candidate.refinement_model}")
        print(f"   Prompt: {candidate.prompt_text[:100]}...")

if __name__ == '__main__':
    main()
