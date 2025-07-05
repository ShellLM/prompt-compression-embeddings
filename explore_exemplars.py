#!/usr/bin/env python3
"""
Explore LLM logs database to find semantically dense exemplars for compression testing.
Uses fast models (gemini-flash, gemini-flash-lite) to analyze and categorize content.
"""

import sqlite3
import json
import subprocess
import argparse
from typing import List, Dict, Tuple
from dataclasses import dataclass
import re

@dataclass
class Exemplar:
    """Represents a potential exemplar for compression testing."""
    id: str
    model: str
    prompt: str
    response: str
    datetime_utc: str
    prompt_length: int
    response_length: int
    density_score: float
    category: str
    reason: str

class ExemplarFinder:
    """Find semantically dense exemplars from the logs database."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        
    def get_candidates(self, min_prompt_length: int = 100, max_prompt_length: int = 5000, 
                      limit: int = 200) -> List[Tuple]:
        """Get candidate responses from database based on length criteria."""
        conn = sqlite3.connect(self.db_path)
        
        # Get diverse samples across different models and time periods
        query = """
        SELECT id, model, prompt, response, datetime_utc, 
               length(prompt) as prompt_len, length(response) as response_len
        FROM responses 
        WHERE prompt_len >= ? AND prompt_len <= ?
        AND prompt IS NOT NULL AND prompt != ''
        AND response IS NOT NULL AND response != ''
        ORDER BY RANDOM()
        LIMIT ?
        """
        
        cursor = conn.execute(query, (min_prompt_length, max_prompt_length, limit))
        candidates = cursor.fetchall()
        conn.close()
        
        return candidates
    
    def analyze_density(self, text: str, model: str = "gemini-flash-lite") -> Tuple[float, str, str]:
        """Analyze semantic density of text using fast LLM."""
        
        analysis_prompt = f"""
        Analyze this text for semantic density - how much meaningful information per word.
        
        Rate from 1-10 where:
        1-3 = Low density (lots of fluff, repetitive, simple)
        4-6 = Medium density (normal conversational content)
        7-8 = High density (technical, complex concepts, rich information)
        9-10 = Very high density (compressed technical content, academic, code)
        
        Also categorize the content type and give a brief reason for the density score.
        
        Respond in JSON format:
        {{
            "density_score": <1-10>,
            "category": "<category_name>",
            "reason": "<brief_explanation>"
        }}
        
        Text to analyze:
        {text[:1000]}...
        """
        
        try:
            result = subprocess.run([
                'llm', '-m', model, analysis_prompt
            ], capture_output=True, text=True, check=True)
            
            # Extract JSON from response
            response_text = result.stdout.strip()
            # Try to find JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                analysis = json.loads(json_str)
                return (
                    float(analysis.get('density_score', 0)),
                    analysis.get('category', 'unknown'),
                    analysis.get('reason', 'no reason provided')
                )
            else:
                return (0.0, 'error', 'failed to parse JSON response')
                
        except Exception as e:
            return (0.0, 'error', f'analysis failed: {str(e)}')
    
    def find_exemplars(self, target_count: int = 100, min_density: float = 6.0) -> List[Exemplar]:
        """Find high-density exemplars for compression testing."""
        
        print(f"Getting candidates from database...")
        candidates = self.get_candidates(limit=300)  # Get more candidates than needed
        print(f"Found {len(candidates)} candidates")
        
        exemplars = []
        
        for i, (id, model, prompt, response, datetime_utc, prompt_len, response_len) in enumerate(candidates):
            if len(exemplars) >= target_count:
                break
                
            print(f"Analyzing candidate {i+1}/{len(candidates)}: {id}")
            
            # Analyze prompt density
            density_score, category, reason = self.analyze_density(prompt)
            
            if density_score >= min_density:
                exemplars.append(Exemplar(
                    id=id,
                    model=model,
                    prompt=prompt,
                    response=response,
                    datetime_utc=datetime_utc,
                    prompt_length=prompt_len,
                    response_length=response_len,
                    density_score=density_score,
                    category=category,
                    reason=reason
                ))
                print(f"  ✓ Added: density={density_score:.1f}, category={category}")
            else:
                print(f"  ✗ Skipped: density={density_score:.1f}, category={category}")
        
        # Sort by density score (highest first)
        exemplars.sort(key=lambda x: x.density_score, reverse=True)
        
        return exemplars
    
    def save_exemplars(self, exemplars: List[Exemplar], output_file: str):
        """Save exemplars to JSON file."""
        data = []
        for exemplar in exemplars:
            data.append({
                'id': exemplar.id,
                'model': exemplar.model,
                'prompt': exemplar.prompt,
                'response': exemplar.response,
                'datetime_utc': exemplar.datetime_utc,
                'prompt_length': exemplar.prompt_length,
                'response_length': exemplar.response_length,
                'density_score': exemplar.density_score,
                'category': exemplar.category,
                'reason': exemplar.reason
            })
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(exemplars)} exemplars to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='Find semantically dense exemplars for compression testing')
    parser.add_argument('--db-path', default='~/.config/io.datasette.llm/logs.db',
                      help='Path to LLM logs database')
    parser.add_argument('--count', type=int, default=100,
                      help='Target number of exemplars to find')
    parser.add_argument('--min-density', type=float, default=6.0,
                      help='Minimum density score (1-10)')
    parser.add_argument('--output', default='exemplars.json',
                      help='Output file for exemplars')
    parser.add_argument('--model', default='gemini-flash-lite',
                      help='Model to use for density analysis')
    
    args = parser.parse_args()
    
    # Expand tilde in db path
    import os
    db_path = os.path.expanduser(args.db_path)
    
    finder = ExemplarFinder(db_path)
    exemplars = finder.find_exemplars(args.count, args.min_density)
    
    print(f"\nFound {len(exemplars)} high-density exemplars")
    
    # Print summary by category
    categories = {}
    for exemplar in exemplars:
        if exemplar.category not in categories:
            categories[exemplar.category] = []
        categories[exemplar.category].append(exemplar)
    
    print("\nCategory breakdown:")
    for category, items in categories.items():
        avg_density = sum(e.density_score for e in items) / len(items)
        print(f"  {category}: {len(items)} exemplars, avg density {avg_density:.1f}")
    
    finder.save_exemplars(exemplars, args.output)

if __name__ == '__main__':
    main()
