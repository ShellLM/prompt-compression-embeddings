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

Coming soon...

## Test Data

Initial test cases extracted from LLM conversation logs with varying complexity:
- Short prompts (~280 chars)
- Medium prompts (~800 chars)  
- Long prompts (~3300 chars)

