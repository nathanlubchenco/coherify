# HaluEval: A Hallucination Evaluation Benchmark for LLMs

*Source: https://github.com/RUCAIBox/HaluEval*

Paper: [HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models](https://arxiv.org/abs/2305.11747)

## Overview

HaluEval is a comprehensive benchmark for evaluating hallucinations in large language models, containing:

- **5,000 general user queries** with ChatGPT responses
- **30,000 task-specific examples** from three tasks:
  - Question answering
  - Knowledge-grounded dialogue
  - Text summarization

## Dataset Construction

### General User Queries
- Based on 52K instruction tuning dataset from [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- Samples three ChatGPT responses per query
- Retains queries with low-similarity responses for human labeling
- Focuses on queries where LLMs are most likely to hallucinate

### Task-Specific Examples
Uses an automatic approach to generate hallucinated samples:

1. **Seed Data**: Based on existing datasets (e.g., HotpotQA)
2. **Generation Methods**:
   - One-pass generation
   - Conversational generation
3. **Filtering**: Uses ChatGPT with ground-truth examples to select plausible and difficult hallucinated samples

## Data Format

Each example contains:
- **Input**: The query or prompt
- **Response**: The model's response
- **Label**: Whether the response contains hallucinations
- **Evidence**: Ground truth information (for task-specific examples)

## Evaluation Tasks

### 1. Question Answering (QA)
Based on HotpotQA dataset with multi-hop reasoning questions.

### 2. Knowledge-Grounded Dialogue
Evaluates hallucinations in conversational contexts with knowledge constraints.

### 3. Text Summarization
Tests whether models hallucinate when summarizing documents.

## Repository Structure

```
HaluEval/
├── data/              # 35K evaluation examples
├── generation/        # Code for data generation
├── evaluation/        # Model evaluation code
└── analysis/          # Analysis tools
```

## Usage

The benchmark can be used to:
- Evaluate hallucination rates in LLMs
- Compare different models' tendency to hallucinate
- Test hallucination detection methods
- Train models to reduce hallucinations

## Integration with Coherify

HaluEval can be integrated with Coherify's coherence measures to:
- Detect hallucinations through coherence analysis
- Compare coherence-based vs. traditional hallucination detection
- Enhance hallucination detection with multi-response coherence checking

## Citation

```bibtex
@article{li2023halueval,
  title={HaluEval: A Large-Scale Hallucination Evaluation Benchmark for Large Language Models},
  author={Li, Junyi and Chen, Xiaoxue and Hovy, Eduard and Jurafsky, Dan},
  journal={arXiv preprint arXiv:2305.11747},
  year={2023}
}
```