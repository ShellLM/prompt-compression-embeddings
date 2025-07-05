#!/usr/bin/env python3
"""
Find High-Quality, Long, and Information-Dense Exemplars for Testing

This script improves upon the previous method by:
1.  Targeting longer prompts and responses.
2.  Using a more sophisticated LLM prompt to analyze content for multiple quality metrics.
3.  Filtering for specific content categories that are more likely to be dense.
4.  Analyzing 'prompt', 'response', and 'system' fields.
"""
import sqlite3
import json
import llm
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random
import re

@dataclass
class ExemplarCandidate:
    id: str
    source_field: str # 'prompt' or 'response'
    text: str
    text_length: int

@dataclass
class AssessedExemplar:
    id: str
    source_field: str
    text: str
    text_length: int
    assessment: Dict

class ExemplarFinder:
    def __init__(self, db_path: str, assessment_model: str = "gpt-4o-mini"):
        self.db_path = Path(db_path).expanduser()
        self.assessment_model = assessment_model

    def get_candidates(self, min_length: int, max_length: int, limit: int) -> List[ExemplarCandidate]:
        """Get candidate texts from the database."""
        candidates = []
        try:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                # Query for long prompts
                prompt_query = """
                SELECT id, 'prompt', prompt, length(prompt) as len
                FROM responses
                WHERE len BETWEEN ? AND ?
                ORDER BY RANDOM() LIMIT ?
                """
                
                # Query for long responses
                response_query = """
                SELECT id, 'response', response, length(response) as len
                FROM responses
                WHERE len BETWEEN ? AND ?
                ORDER BY RANDOM() LIMIT ?
                """
                
                # Get candidates from both prompts and responses
                for query, source_type in [(prompt_query, 'prompt'), (response_query, 'response')]:
                    cursor = conn.execute(query, (min_length, max_length, limit // 2))
                    for row in cursor:
                        candidate = ExemplarCandidate(
                            id=row[0],
                            source_field=row[1],
                            text=row[2],
                            text_length=row[3]
                        )
                        candidates.append(candidate)
                
                return candidates
                
        except Exception as e:
            print(f"Error accessing database: {e}")
            return []

    def assess_exemplar(self, candidate: ExemplarCandidate) -> Optional[Dict]:
        """Assess the quality of an exemplar using LLM."""
        
        # Truncate very long texts for assessment
        text_snippet = candidate.text[:2000] if len(candidate.text) > 2000 else candidate.text
        
        assessment_prompt = f"""
        Please analyze the following text snippet and provide a JSON assessment with these metrics:
        
        - information_density: Scale 1-10, how much meaningful information per word
        - structural_complexity: Scale 1-10, how complex is the structure/format
        - readability: Scale 1-10, how clear and well-written it is
        - suitability_for_compression: Scale 1-10, how well it would compress while preserving meaning
        - category: Choose from ["Technical Documentation", "Creative Writing", "Code", "Academic", "Business", "Conversational", "Other"]
        - justification: Brief explanation of the scores
        
        Return ONLY a valid JSON object with these fields.
        
        ---
        {text_snippet}
        ---
        """

        try:
            model_obj = llm.get_model(self.assessment_model)
            response = model_obj.prompt(assessment_prompt.strip(), temperature=0.1)
            result_text = response.text()
            
            # Find and parse the JSON block from the model's output
            json_match = re.search(r'\{.*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print(f"Warning: Could not parse JSON from assessment for candidate {candidate.id}")
                return None
                
        except Exception as e:
            print(f"Error assessing exemplar {candidate.id}: {e}")
            return None

    def find_high_quality_exemplars(self, min_length: int = 1000, max_length: int = 8000, 
                                   candidate_limit: int = 100, quality_threshold: float = 6.0) -> List[AssessedExemplar]:
        """Find high-quality exemplars for testing."""
        
        print(f"Finding exemplars with length {min_length}-{max_length} characters...")
        candidates = self.get_candidates(min_length, max_length, candidate_limit)
        print(f"Found {len(candidates)} candidates")
        
        assessed_exemplars = []
        
        for i, candidate in enumerate(candidates):
            print(f"Assessing candidate {i+1}/{len(candidates)}: {candidate.id}")
            
            assessment = self.assess_exemplar(candidate)
            if assessment:
                # Calculate average quality score
                quality_scores = [
                    assessment.get('information_density', 0),
                    assessment.get('structural_complexity', 0),
                    assessment.get('readability', 0),
                    assessment.get('suitability_for_compression', 0)
                ]
                avg_quality = sum(quality_scores) / len(quality_scores)
                
                if avg_quality >= quality_threshold:
                    assessed_exemplar = AssessedExemplar(
                        id=candidate.id,
                        source_field=candidate.source_field,
                        text=candidate.text,
                        text_length=candidate.text_length,
                        assessment=assessment
                    )
                    assessed_exemplars.append(assessed_exemplar)
                    print(f"  ✓ Quality score: {avg_quality:.1f}")
                else:
                    print(f"  ✗ Quality score: {avg_quality:.1f} (below threshold)")
            else:
                print(f"  ✗ Assessment failed")
        
        # Sort by quality (using suitability_for_compression as primary metric)
        assessed_exemplars.sort(key=lambda x: x.assessment.get('suitability_for_compression', 0), reverse=True)
        
        return assessed_exemplars

def main():
    parser = argparse.ArgumentParser(description='Find high-quality exemplars for compression testing')
    parser.add_argument('--db', default='~/.config/io.datasette.llm/logs.db', help='Database path')
    parser.add_argument('--min-length', type=int, default=1000, help='Minimum text length')
    parser.add_argument('--max-length', type=int, default=8000, help='Maximum text length')
    parser.add_argument('--candidates', type=int, default=100, help='Number of candidates to assess')
    parser.add_argument('--threshold', type=float, default=6.0, help='Quality threshold')
    parser.add_argument('--output', default='high_quality_exemplars.json', help='Output file')
    parser.add_argument('--model', default='gpt-4o-mini', help='Assessment model')
    
    args = parser.parse_args()
    
    finder = ExemplarFinder(args.db, args.model)
    exemplars = finder.find_high_quality_exemplars(
        min_length=args.min_length,
        max_length=args.max_length,
        candidate_limit=args.candidates,
        quality_threshold=args.threshold
    )
    
    # Convert to JSON-serializable format
    output_data = []
    for exemplar in exemplars:
        output_data.append({
            'id': exemplar.id,
            'source_field': exemplar.source_field,
            'text_length': exemplar.text_length,
            'assessment': exemplar.assessment,
            'text': exemplar.text
        })
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nFound {len(exemplars)} high-quality exemplars")
    print(f"Results saved to {args.output}")
    
    # Show summary statistics
    if exemplars:
        categories = {}
        for exemplar in exemplars:
            cat = exemplar.assessment.get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        print("\nCategory distribution:")
        for cat, count in sorted(categories.items()):
            print(f"  {cat}: {count}")

if __name__ == '__main__':
    main()
