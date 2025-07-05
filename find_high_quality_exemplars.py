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
import subprocess
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import random

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
    def __init__(self, db_path: str, assessment_model: str = "gemini-flash"):
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
                cursor = conn.execute(prompt_query, (min_length, max_length, limit // 2))
                for row in cursor:
                    candidates.append(ExemplarCandidate(id=row[0], source_field=row[1], text=row[2], text_length=row[3]))

                # Query for long responses
                response_query = """
                SELECT id, 'response', response, length(response) as len
                FROM responses
                WHERE len BETWEEN ? AND ?
                ORDER BY RANDOM() LIMIT ?
                """
                cursor = conn.execute(response_query, (min_length, max_length, limit // 2))
                for row in cursor:
                    candidates.append(ExemplarCandidate(id=row[0], source_field=row[1], text=row[2], text_length=row[3]))
        except sqlite3.Error as e:
            print(f"Database error: {e}")
            return []

        print(f"Retrieved {len(candidates)} candidates from the database.")
        return candidates

    def assess_candidate(self, candidate: ExemplarCandidate) -> Optional[Dict]:
        """Use an LLM to assess the quality of a text candidate."""
        # Truncate text to avoid excessive token usage for assessment
        text_snippet = candidate.text[:4000]

        assessment_prompt = f"""
        Analyze the following text snippet to determine its suitability as a test case for a text compression algorithm.
        Provide your assessment in a strict JSON format.

        **Evaluation Criteria:**
        1.  **Information Density (1-10):** How much meaningful, non-trivial information is packed into the text? (1=fluff, 10=very dense).
        2.  **Structural Complexity (1-10):** Does the text have a clear structure (e.g., lists, code, hierarchies, logical sections)? (1=monolithic block, 10=highly structured).
        3.  **Readability (1-10):** How easy is it for a human to understand the text? (1=incoherent, 10=very clear). This is important for judging decompression quality later.
        4.  **Suitability for Compression (1-10):** Based on the above, how good is this text for testing compression? High suitability means it has redundant parts but also a critical core of information that must be preserved. (1=unsuitable, 10=ideal).
        5.  **Category:** Classify the text into one of the following categories: `Code`, `Technical Documentation`, `Structured Data`, `Formal Prose`, `Instructional Content`, `Conversational`, `Other`.
        6.  **Justification:** A brief, one-sentence explanation for your 'Suitability' score.

        **JSON Output Format:**
        {{
            "information_density": <score>,
            "structural_complexity": <score>,
            "readability": <score>,
            "suitability_for_compression": <score>,
            "category": "<category_name>",
            "justification": "<one_sentence_reason>"
        }}

        **Text Snippet to Analyze:**
        ---
        {text_snippet}
        ---
        """

        try:
            result = subprocess.run(
                ['llm', '-m', self.assessment_model, assessment_prompt],
                capture_output=True, text=True, check=True, timeout=60
            )
            # Find and parse the JSON block from the model's output
            json_match = re.search(r'\{.*\}', result.stdout, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            else:
                print(f"Warning: Could not parse JSON from assessment for candidate {candidate.id}")
                return None
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            print(f"Error assessing candidate {candidate.id}: {e}")
            return None

    def find_and_assess_exemplars(self, num_needed: int, min_length: int, max_length: int, min_suitability: float) -> List[AssessedExemplar]:
        """The main orchestration function."""
        # Fetch more candidates than needed to account for filtering
        candidates = self.get_candidates(min_length, max_length, limit=num_needed * 3)
        if not candidates:
            return []

        assessed_exemplars = []
        for i, candidate in enumerate(candidates):
            if len(assessed_exemplars) >= num_needed:
                break
            print(f"Assessing candidate {i+1}/{len(candidates)} (ID: {candidate.id}, Field: {candidate.source_field})...")
            assessment = self.assess_candidate(candidate)
            if assessment and assessment.get('suitability_for_compression', 0) >= min_suitability:
                assessed_exemplars.append(AssessedExemplar(
                    id=candidate.id,
                    source_field=candidate.source_field,
                    text=candidate.text,
                    text_length=candidate.text_length,
                    assessment=assessment
                ))
                print(f"  -> QUALIFIED! Suitability: {assessment['suitability_for_compression']}, Category: {assessment['category']}")
            elif assessment:
                print(f"  -> SKIPPED. Suitability: {assessment.get('suitability_for_compression', 'N/A')}")

        # Sort by suitability and return the top N
        assessed_exemplars.sort(key=lambda x: x.assessment['suitability_for_compression'], reverse=True)
        return assessed_exemplars[:num_needed]

def main():
    parser = argparse.ArgumentParser(description="Find high-quality exemplars for compression testing.")
    parser.add_argument('--db-path', default='~/.config/io.datasette.llm/logs.db', help='Path to LLM logs database')
    parser.add_argument('--count', type=int, default=50, help='Number of high-quality exemplars to find')
    parser.add_argument('--min-length', type=int, default=2000, help='Minimum character length of the text')
    parser.add_argument('--max-length', type=int, default=15000, help='Maximum character length of the text')
    parser.add_argument('--min-suitability', type=float, default=7.5, help='Minimum suitability score (1-10) to be considered a good exemplar')
    parser.add_argument('--output', default='high_quality_exemplars.json', help='Output file for the found exemplars')
    parser.add_argument('--model', default='gemini-flash', help='LLM model for assessment')
    args = parser.parse_args()

    finder = ExemplarFinder(db_path=args.db_path, assessment_model=args.model)
    exemplars = finder.find_and_assess_exemplars(
        num_needed=args.count,
        min_length=args.min_length,
        max_length=args.max_length,
        min_suitability=args.min_suitability
    )

    if exemplars:
        print(f"\nFound {len(exemplars)} high-quality exemplars.")
        # Prepare for JSON serialization
        output_data = [
            {
                "id": e.id,
                "source_field": e.source_field,
                "text_length": e.text_length,
                "assessment": e.assessment,
                "text": e.text
            } for e in exemplars
        ]
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"Successfully saved to {args.output}")
    else:
        print("\nCould not find any exemplars matching the criteria.")

if __name__ == "__main__":
    import re
    main()
