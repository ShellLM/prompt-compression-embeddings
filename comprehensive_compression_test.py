import llm
import sqlite3
import json
import os
from datetime import datetime
import time
from tqdm import tqdm
import re
import numpy as np

# --- CONFIGURATION ---
DB_PATH = os.path.expanduser('~/.config/io.datasette.llm/logs.db')
EXEMPLARS_PATH = 'high_quality_exemplars.json'
RESULTS_PATH = 'new_comprehensive_results.json'

COMPRESSION_MODELS = [
    "claude-3.7-sonnet",
    "gpt-4o",
    "gpt-4o-mini"
]
DECOMPRESSION_MODELS = [
    "claude-3.7-sonnet",
    "gpt-4o",
    "gpt-4o-mini"
]
EMBEDDING_MODEL = "text-embedding-3-small"

# --- STATIC ONE-SHOT EXEMPLAR ---
# This is a fixed example to guide the compression model.
EXEMPLAR_PROMPT = "Can you write a short, informal email to my team about the new coffee machine? Mention it's a De'Longhi, it's in the main kitchen, and that the company paid for it so it's free to use. Also, remind them to clean the milk frother after use."
EXEMPLAR_RESPONSE = "Subject: New Coffee Machine!\n\nHi Team,\n\nJust a quick note to let you know we have a brand new De'Longhi coffee machine in the main kitchen.\n\nPlease feel free to use it - it's on the house! \n\nOne small request: if you use the milk frother, please make sure to clean it out afterwards so it's ready for the next person.\n\nEnjoy!\n\nBest,"
EXEMPLAR_COMPRESSED = {
  "type": "email",
  "to": "team",
  "subject": "New coffee machine",
  "tone": "informal",
  "points": [
    "what: De'Longhi coffee machine",
    "where: main kitchen",
    "cost: free (company expense)",
    "action_required: clean milk frother after use"
  ]
}

# --- HELPER FUNCTIONS ---

def get_db_connection():
    return sqlite3.connect(DB_PATH)

def get_response_by_id(conn, response_id):
    cursor = conn.cursor()
    cursor.execute("SELECT prompt, response FROM responses WHERE id = ?", (response_id,))
    return cursor.fetchone()

def load_exemplars(path):
    with open(path, 'r') as f:
        return json.load(f)

def clean_json_string(s):
    s = s.strip()
    # Handle JSON that might be embedded in markdown code blocks
    if s.startswith("json"):
        s = s[7:]
    if s.endswith(""):
        s = s[:-3]
    s = s.strip()
    # Find JSON objects using a simpler approach
    # Look for balanced braces starting from the first {
    if '{' in s:
        start = s.find('{')
        brace_count = 0
        for i, char in enumerate(s[start:], start):
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0:
                    return s[start:i+1]
    return s

def calculate_token_count(text, model):
    try:
        return llm.get_model(model).count_tokens(text)
    except Exception:
        return len(text) // 4

def get_embedding_score(original_prompt, original_response, decompressed_prompt, decompressed_response):
    try:
        embedding_model = llm.get_embedding_model(EMBEDDING_MODEL)
        original_text = f"PROMPT:\n{original_prompt}\n\nRESPONSE:\n{original_response}"
        decompressed_text = f"PROMPT:\n{decompressed_prompt}\n\nRESPONSE:\n{decompressed_response}"
        
        # Guard against empty text, which can cause embedding errors
        if not original_text.strip() or not decompressed_text.strip():
            return 0.0

        original_embedding = embedding_model.embed(original_text)
        decompressed_embedding = embedding_model.embed(decompressed_text)

        vec1 = np.array(original_embedding)
        vec2 = np.array(decompressed_embedding)

        if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
            return 0.0

        cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        return float(cosine_similarity)
    except Exception as e:
        print(f"Error calculating embedding score: {e}")
        return 0.0

def compress_prompt(model, original_prompt, original_response):
    full_prompt = f"""You are an expert prompt compressor. Your task is to significantly compress the following prompt-response pair into a compact JSON format.

Here is an example of ideal compression:
---
[Exemplar]
Original Prompt:
{EXEMPLAR_PROMPT}

Original Response:
{EXEMPLAR_RESPONSE}

Compressed JSON:
{json.dumps(EXEMPLAR_COMPRESSED, indent=2)}
---

Now, compress the following prompt-response pair in the same way. The compressed version MUST be a single, valid JSON object only.

[User Request]
Original Prompt:
{original_prompt}

Original Response:
{original_response}

Compressed JSON:
"""
    try:
        response = llm.get_model(model).prompt(full_prompt, temperature=0.0).text()
        return response.strip()
    except Exception as e:
        print(f"Error during compression with {model}: {e}")
        return "{'error': 'API call failed'}"

def decompress_prompt(model, compressed_json, original_prompt):
    decompression_instruction = f"""You are an expert prompt decompressor. Reconstruct the original prompt from this compressed JSON object. The goal is to recreate the user's original request as faithfully as possible.

Compressed JSON:
{compressed_json}

Use the following original prompt only as a high-level guide for the topic and structure of the expected output, but DO NOT simply copy it. Your primary instruction is to decompress the JSON.

Original Prompt for context:
{original_prompt}

Reconstructed Prompt:
"""
    try:
        response = llm.get_model(model).prompt(decompression_instruction, temperature=0.0).text()
        return response.strip()
    except Exception as e:
        print(f"Error during decompression with {model}: {e}")
        return f"Decompression failed: {e}"

def main():
    conn = get_db_connection()
    exemplars = load_exemplars(EXEMPLARS_PATH)
    results = []

    print("Starting comprehensive prompt compression test...")
    
    total_iterations = len(exemplars) * len(COMPRESSION_MODELS) * len(DECOMPRESSION_MODELS)
    pbar = tqdm(total=total_iterations, desc="Testing Combinations")

    for exemplar in exemplars:
        try:
            original_data = get_response_by_id(conn, exemplar['id'])
            if not original_data:
                pbar.update(len(COMPRESSION_MODELS) * len(DECOMPRESSION_MODELS))
                continue

            original_prompt, original_response = original_data
            
            # The one-shot example is now static and defined above.
            # No need to load a second exemplar from the file.

            for comp_model in COMPRESSION_MODELS:
                start_time = time.time()
                compressed_output = compress_prompt(comp_model, original_prompt, original_response)
                compression_time = time.time() - start_time
                
                try:
                    compressed_json_str = clean_json_string(compressed_output)
                    compressed_json = json.loads(compressed_json_str)
                except json.JSONDecodeError:
                    compressed_json = {"error": "Invalid JSON produced", "raw": compressed_output}

                for decomp_model in DECOMPRESSION_MODELS:
                    start_time = time.time()
                    decompressed_prompt = decompress_prompt(decomp_model, json.dumps(compressed_json), original_prompt)
                    decompression_time = time.time() - start_time
                    
                    decompressed_response = "NOT_TESTED"
                    
                    original_tokens = calculate_token_count(original_prompt, comp_model)
                    compressed_tokens = calculate_token_count(json.dumps(compressed_json), comp_model)
                    compression_ratio = original_tokens / compressed_tokens if compressed_tokens > 0 else 0
                    
                    embedding_score = get_embedding_score(
                        original_prompt, original_response, 
                        decompressed_prompt, decompressed_response
                    )
                    
                    result_entry = {
                        "exemplar_id": exemplar['id'],
                        "compression_model": comp_model,
                        "decompression_model": decomp_model,
                        "original_prompt_tokens": original_tokens,
                        "compressed_json_tokens": compressed_tokens,
                        "compression_ratio": compression_ratio,
                        "embedding_similarity": embedding_score,
                        "compression_time_s": compression_time,
                        "decompression_time_s": decompression_time,
                        "compressed_json": compressed_json
                    }
                    results.append(result_entry)
                    pbar.update(1)

        except Exception as e:
            print(f"\nError processing exemplar {exemplar.get('id', 'N/A')}: {e}")
            pbar.update(len(COMPRESSION_MODELS) * len(DECOMPRESSION_MODELS))

    pbar.close()
    conn.close()

    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nComprehensive test finished. Results saved to {RESULTS_PATH}")
    
    if results:
        best_ratio = max([r for r in results if r['compression_ratio'] is not None], key=lambda x: x['compression_ratio'], default=None)
        best_similarity = max([r for r in results if r['embedding_similarity'] is not None], key=lambda x: x['embedding_similarity'], default=None)
        if best_ratio:
            print(f"\nBest Compression Ratio: {best_ratio['compression_ratio']:.2f} (by {best_ratio['compression_model']})")
        if best_similarity:
            print(f"Best Embedding Similarity: {best_similarity['embedding_similarity']:.4f} (decompress by {best_similarity['decompression_model']})")

if __name__ == "__main__":
    main()
