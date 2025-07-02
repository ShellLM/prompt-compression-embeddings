# This will be inserted into the main script

from token_counter import TokenCounter

@dataclass
class CompressionResult:
    """Results from a compression attempt."""
    original_prompt: str
    compressed_prompt: str
    compression_strategy: str
    original_length: int
    compressed_length: int
    compression_ratio: float
    original_tokens: int
    compressed_tokens: int
    token_compression_ratio: float
    original_embedding: List[float]
    compressed_embedding: List[float]
    similarity_score: float

# Updated test function
async def run_compression_test(test_case: TestCase, strategy: str, 
                             compressor: PromptCompressor, 
                             embedder: EmbeddingComparer) -> CompressionResult:
    """Run a single compression test."""
    print(f"Testing {test_case.id} with strategy '{strategy}'...")
    
    # Compress the prompt
    compressed = compressor.compress_prompt(test_case.prompt, strategy)
    
    # Get embeddings for both versions
    original_embedding = embedder.get_embedding(test_case.prompt)
    compressed_embedding = embedder.get_embedding(compressed)
    
    # Calculate similarity
    similarity = embedder.cosine_similarity(original_embedding, compressed_embedding)
    
    # Calculate compression metrics
    original_len = len(test_case.prompt)
    compressed_len = len(compressed)
    compression_ratio = compressed_len / original_len if original_len > 0 else 0
    
    # Calculate token metrics
    original_tokens = TokenCounter.get_best_estimate(test_case.prompt)
    compressed_tokens = TokenCounter.get_best_estimate(compressed)
    token_compression_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 0
    
    return CompressionResult(
        original_prompt=test_case.prompt,
        compressed_prompt=compressed,
        compression_strategy=strategy,
        original_length=original_len,
        compressed_length=compressed_len,
        compression_ratio=compression_ratio,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        token_compression_ratio=token_compression_ratio,
        original_embedding=original_embedding,
        compressed_embedding=compressed_embedding,
        similarity_score=similarity
    )
