import llm
import json
from typing import List

def get_embedding(text: str, embedding_model: str = "text-embedding-3-small") -> List[float]:
    """Get embedding for text using llm Python API."""
    try:
        # Use the llm Python API directly
        embedding_model_obj = llm.get_embedding_model(embedding_model)
        embedding_data = embedding_model_obj.embed(text)
        return embedding_data
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return []

# Example usage
if __name__ == "__main__":
    test_text = "This is a test sentence for embedding."
    result = get_embedding(test_text)
    print(f"Embedding length: {len(result)}")
    print(f"First 5 values: {result[:5]}")
