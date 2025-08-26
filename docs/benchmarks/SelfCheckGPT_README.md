# SelfCheckGPT

*Source: https://github.com/potsawee/selfcheckgpt*

[![arxiv](https://img.shields.io/badge/arXiv-2303.08896-b31b1b.svg)](https://arxiv.org/abs/2303.08896)
[![PyPI version selfcheckgpt](https://badge.fury.io/py/selfcheckgpt.svg?kill_cache=1)](https://pypi.python.org/pypi/selfcheckgpt/)
[![Downloads](https://pepy.tech/badge/selfcheckgpt)](https://pepy.tech/project/selfcheckgpt)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Project page for the paper "[SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896)"

## Overview

SelfCheckGPT is a method for detecting hallucinations in large language model outputs without requiring external resources. It works by sampling multiple responses from the same model and checking consistency between them.

### Key Features

- **Zero-resource**: No external databases or knowledge sources required
- **Black-box**: Works with any LLM without access to internal probabilities
- **Multiple methods**: BERTScore, Question-Answering, n-gram, NLI, and LLM-Prompting variants

## Methods

### 1. SelfCheck-BERTScore
Uses BERTScore to measure semantic similarity between sampled passages.

### 2. SelfCheck-MQAG (Question Answering)
Generates questions from sentences and checks if sampled passages can answer them consistently.

### 3. SelfCheck-Ngram
Compares n-gram probabilities across sampled passages.

### 4. SelfCheck-NLI (Recommended)
Uses Natural Language Inference to check if sentences are entailed by sampled passages. Uses DeBERTa-v3-large fine-tuned on Multi-NLI.

### 5. SelfCheck-Prompt
Prompts an LLM (GPT-3.5, Llama2, Mistral) to assess consistency in a zero-shot setup.

## Installation

```bash
pip install selfcheckgpt
```

## Basic Usage

```python
from selfcheckgpt.modeling_selfcheck import SelfCheckNLI
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
selfcheck_nli = SelfCheckNLI(device=device)

# Evaluate sentences against sampled passages
sent_scores_nli = selfcheck_nli.predict(
    sentences = sentences,                          
    sampled_passages = [sample1, sample2, sample3], 
)
# Higher scores indicate higher likelihood of hallucination
```

## Dataset

The `wiki_bio_gpt3_hallucination` dataset consists of 238 annotated passages with human-labeled hallucinations at the sentence level.

Available at: https://huggingface.co/datasets/potsawee/wiki_bio_gpt3_hallucination

## Performance Results

On the wiki_bio_gpt3_hallucination dataset:

| Method                                | NonFact (AUC-PR) | Factual (AUC-PR) | Ranking (PCC) |
|---------------------------------------|:----------------:|:----------------:|:-------------:|
| Random Guessing                       |      72.96       |      27.04       |       -       |
| GPT-3 Avg(-logP)                      |      83.21       |      53.97       |     57.04     |
| SelfCheck-BERTScore                   |      81.96       |      44.23       |     58.18     |
| SelfCheck-QA                          |      84.26       |      48.14       |     61.07     |
| SelfCheck-Unigram                     |      85.63       |      58.47       |     64.71     |
| SelfCheck-NLI                         |      92.50       |      66.08       |     74.14     |
| **SelfCheck-Prompt (gpt-3.5-turbo)**  |    **93.42**     |    **67.09**     |   **78.32**   |

## Integration with Coherify

SelfCheckGPT is integrated into Coherify as:
- A benchmark adapter for hallucination detection tasks
- Uses the NLI variant by default for best performance
- Can be combined with coherence measures for enhanced detection

See `/coherify/benchmarks/selfcheckgpt.py` for our implementation.

## Citation

```bibtex
@inproceedings{manakul2023selfcheckgpt,
  title={SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models},
  author={Manakul, Potsawee and Liusie, Adian and Gales, Mark JF},
  booktitle={EMNLP},
  year={2023}
}
```