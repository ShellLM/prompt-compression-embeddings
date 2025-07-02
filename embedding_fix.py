def get_embedding(self, text: str) -> List[float]:
    """Get embedding for text using llm CLI."""
    import json
    
    # Use shell to pipe text to llm embed, suppress debug output
    cmd = f'echo {json.dumps(text)} | llm embed -m {self.embedding_model} 2>/dev/null'
    result = subprocess.run(['sh', '-c', cmd], capture_output=True, text=True, check=True)
    
    # Extract just the JSON array part (the last line should be the JSON)
    output_lines = result.stdout.strip().split('\n')
    json_line = output_lines[-1]  # The JSON array should be the last line
    
    # Parse the JSON array output
    embedding_data = json.loads(json_line)
    return embedding_data
