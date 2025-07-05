#!/usr/bin/env python3
"""
Explore LLM logs database to find semantically dense exemplars for compression testing.
Uses fast models to analyze and categorize content.
"""

import sqlite3
import json
import llm
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
    
    def __init__(self, db_path: str, analysis_model: str = "gpt-4o-mini"):
        self.db_path = db_path
        self.analysis_model = analysis_model
    
    def get_candidates(self, min_length: int = 500, limit: int = 50) -> List[Tuple]:
        """Get candidate prompt-response pairs from database."""
        with sqlite3.connect(self.db_path) as conn:
            query = """
            SELECT id, model, prompt, response, datetime_utc,
                   LENGTH(prompt) as prompt_len, LENGTH(response) as response_len
            FROM responses 
            WHERE prompt_len > ? AND response_len > ?
            ORDER BY RANDOM() 
            LIMIT ?
            """
            cursor = conn.execute(query, (min_length, min_length, limit))
            return cursor.fetchall()
    
    def analyze_density(self, text: str, model: str = None) -> Dict:
        """Analyze semantic density of text using LLM."""
        if model is None:
            model = self.analysis_model
            
        analysis_prompt = f"""
        Analyze this text for semantic density and compression potential. 
        Rate 1-10 on:
        - Information density (how much meaningful info per word)
        - Structural complexity (formatting, lists, code, etc.)
        - Redundancy (repeated concepts/phrases)
        - Compression potential (how well it would compress)
        
        Also categorize as: Technical, Creative, Academic, Business, Code, Conversational, Other
        
        Return JSON only:
        {{
            "density_score": <1-10>,
            "complexity_score": <1-10>, 
            "redundancy_score": <1-10>,
            "compression_score": <1-10>,
            "category": "<category>",
            "reason": "<brief explanation>"
        }}
        
        Text to analyze:
        {text[:1000]}...
        """
        
        try:
            model_obj = llm.get_model(model)
            response = model_obj.prompt(analysis_prompt.strip(), temperature=0.1)
            response_text = response.text().strip()
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print(f"Warning: Could not parse JSON from analysis")
                return {"density_score": 0, "complexity_score": 0, "redundancy_score": 0, 
                       "compression_score": 0, "category": "Unknown", "reason": "Analysis failed"}
                
        except Exception as e:
            print(f"Error analyzing density: {e}")
            return {"density_score": 0, "complexity_score": 0, "redundancy_score": 0, 
                   "compression_score": 0, "category": "Unknown", "reason": f"Error: {e}"}
    
    def find_exemplars(self, min_length: int = 500, limit: int = 50, 
                      density_threshold: float = 6.0) -> List[Exemplar]:
        """Find high-density exemplars suitable for compression testing."""
        
        print(f"Finding exemplars from {limit} candidates...")
        candidates = self.get_candidates(min_length, limit)
        print(f"Analyzing {len(candidates)} candidates...")
        
        exemplars = []
        
        for i, (id, model, prompt, response, datetime_utc, prompt_len, response_len) in enumerate(candidates):
            print(f"Analyzing candidate {i+1}/{len(candidates)}: {id}")
            
            # Analyze both prompt and response
            combined_text = f"PROMPT: {prompt}\n\nRESPONSE: {response}"
            analysis = self.analyze_density(combined_text)
            
            density_score = analysis.get("density_score", 0)
            
            if density_score >= density_threshold:
                exemplar = Exemplar(
                    id=id,
                    model=model,
                    prompt=prompt,
                    response=response,
                    datetime_utc=datetime_utc,
                    prompt_length=prompt_len,
                    response_length=response_len,
                    density_score=density_score,
                    category=analysis.get("category", "Unknown"),
                    reason=analysis.get("reason", "")
                )
                exemplars.append(exemplar)
                print(f"  ✓ Density: {density_score:.1f} ({analysis.get('category', 'Unknown')})")
            else:
                print(f"  ✗ Density: {density_score:.1f} (below threshold)")
        
        # Sort by density score
        exemplars.sort(key=lambda x: x.density_score, reverse=True)
        
        return exemplars

def main():
    parser = argparse.ArgumentParser(description='Find dense exemplars for compression testing')
    parser.add_argument('--db', default='~/.config/io.datasette.llm/logs.db', help='Database path')
    parser.add_argument('--min-length', type=int, default=500, help='Minimum text length')
    parser.add_argument('--limit', type=int, default=50, help='Number of candidates to analyze')
    parser.add_argument('--threshold', type=float, default=6.0, help='Density threshold')
    parser.add_argument('--output', default='exemplars.json', help='Output file')
    parser.add_argument('--model', default='gpt-4o-mini', help='Analysis model')
    
    args = parser.parse_args()
    
    # Expand user path
    import os
    db_path = os.path.expanduser(args.db)
    
    finder = ExemplarFinder(db_path, args.model)
    exemplars = finder.find_exemplars(
        min_length=args.min_length,
        limit=args.limit,
        density_threshold=args.threshold
    )
    
    # Convert to JSON-serializable format
    output_data = []
    for exemplar in exemplars:
        output_data.append({
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
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nFound {len(exemplars)} high-density exemplars")
    print(f"Results saved to {args.output}")
    
    # Show summary
    if exemplars:
        categories = {}
        for exemplar in exemplars:
            cat = exemplar.category
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nCategory distribution:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")
        
        print(f"\nTop 3 exemplars by density:")
        for i, exemplar in enumerate(exemplars[:3]):
            print(f"{i+1}. {exemplar.id} (density: {exemplar.density_score:.1f}, {exemplar.category})")
            print(f"   {exemplar.reason}")

if __name__ == '__main__':
    main()
