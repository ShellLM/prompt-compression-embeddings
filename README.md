# Prompt Compression Embeddings Test

A research project to test and evaluate different prompt compression/summarization methods using embedding similarity comparison.

## Overview

This project takes input prompts and compression instruction prompts to generate compressed summaries, then uses embeddings to measure the semantic similarity between original and compressed versions. The goal is to find optimal compression strategies that maintain semantic meaning while reducing token count.

## Features

- Extract test data from LLM conversation logs
- Generate compressed versions of prompts using various compression strategies
- Compare semantic similarity using Jina embeddings v4
- Parallel processing for efficiency
- Systematic evaluation of different compression approaches

## Requirements

- simonw's llm CLI tool
- Jina embeddings v4 (via llm)
- SQLite database access to LLM logs

## Usage

This project is run in two main phases:

### 1. Find High-Quality Test Data

First, you need to mine your `llm` logs database for good test examples ("exemplars"). This script finds long, information-dense texts and uses an LLM to assess their quality for testing compression.

```bash
# This will search your llm logs for 50 exemplars with a length
# between 2000 and 15000 characters and a suitability score of at least 7.5.
# The results are saved to high_quality_exemplars.json.

./find_high_quality_exemplars.py --count 50 --min-length 2000 --min-suitability 7.5
```

You can adjust the parameters as needed. Check the script for more options:
```bash
./find_high_quality_exemplars.py --help
```

### 2. Run the Comprehensive Compression Test

Once you have your `high_quality_exemplars.json` file, you can run the main testing suite. This script will test a matrix of different models and compression strategies on the exemplars.

```bash
# This will use the generated exemplars and run the full test suite,
# saving the output to comprehensive_results.json.

./comprehensive_compression_test.py --exemplars high_quality_exemplars.json --output comprehensive_results.json
```

This process can take a long time and will make many LLM calls. The results will be saved incrementally.

### 3. (Optional) Explore Results

The output `comprehensive_results.json` is a detailed file containing all test data. You can manually inspect this file or use data analysis tools to explore which compression strategies and models performed best.
## Test Data

Initial test cases extracted from LLM conversation logs with varying complexity:
- Short prompts (~280 chars)
- Medium prompts (~800 chars)  
- Long prompts (~3300 chars)

