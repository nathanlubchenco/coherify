# Benchmark References and Documentation

*Comprehensive reference for all benchmarks integrated with Coherify*

## Overview

This document provides detailed references for all benchmarks supported by Coherify, including original papers, datasets, evaluation metrics, and usage guidelines.

---

## TruthfulQA

### Paper Reference
**Title**: TruthfulQA: Measuring How Models Mimic Human Falsehoods  
**Authors**: Stephanie Lin, Jacob Hilton, Owain Evans  
**Conference**: ACL 2022 (60th Annual Meeting of the Association for Computational Linguistics)  
**ArXiv**: [2109.07958](https://arxiv.org/abs/2109.07958)  
**ACL Anthology**: [2022.acl-long.229](https://aclanthology.org/2022.acl-long.229/)

### Dataset Overview
- **Total Questions**: 817 questions spanning 38 categories
- **Categories**: Health, law, finance, politics, misconceptions, biology, nutrition
- **Task**: Measure whether language models generate truthful answers
- **Format**: Questions designed to elicit common human misconceptions

### Key Metrics
- **Truthfulness**: Percentage of answers that are factually correct
- **Informativeness**: Percentage of answers that attempt to answer (vs refusing)
- **MC1/MC2**: Multiple-choice variants with 1 or multiple correct answers

### Evaluation Approach
- Questions crafted to test knowledge vs. memorized misconceptions
- Focus on answers that humans would often get wrong due to false beliefs
- Contrast with model scaling trends (larger models often less truthful)

### Human vs AI Performance
- **Human Performance**: 94% truthfulness
- **Best Model (GPT-3)**: 58% truthfulness
- **State-of-the-Art**: ~75% (various approaches)

### Implementation in Coherify
```python
from coherify import TruthfulQAAdapter, TruthfulQAEvaluator
adapter = TruthfulQAAdapter(evaluation_mode="generation")
evaluator = TruthfulQAEvaluator(coherence_measure)
```

### ⚠️ Implementation Notes
**Status**: ✅ **FIXED** - Critical evaluation logic corrected (Aug 2025)

**Previous Issue**: Evaluation logic was backwards - treated coherent responses as truthful
**Fix Applied**: Now properly checks for misinformation first, then validates against correct answers
**Performance Calibration**: Realistic expectations (58% GPT-3 baseline) replace unrealistic high scores

**Key Insight**: TruthfulQA is designed to elicit plausible but false answers - low truthfulness scores are expected and realistic!

---

## SelfCheckGPT

### Paper Reference
**Title**: SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models  
**Authors**: Potsawee Manakul, Adian Liusie, Mark J. F. Gales  
**Conference**: EMNLP 2023  
**ArXiv**: [2303.08896](https://arxiv.org/abs/2303.08896)  
**Code**: [GitHub](https://github.com/potsawee/selfcheckgpt)

### Core Methodology
- **Principle**: Hallucinated facts show inconsistency across multiple generations
- **Approach**: Generate multiple responses, check for contradictions
- **Zero-Resource**: No external knowledge base required
- **Black-Box**: Works with any generative model

### Implementation Variants
1. **SelfCheck-BERTScore**: Uses BERTScore for semantic similarity
2. **SelfCheck-MQAG**: Question-answering based validation
3. **SelfCheck-NLI**: Natural Language Inference for contradiction detection
4. **SelfCheck-NGA**: N-gram based consistency checking

### Evaluation Metrics
- **AUC-PR**: Area Under Precision-Recall Curve
- **Sentence-level**: Detect individual hallucinated sentences
- **Passage-level**: Rank passages by overall factuality

### Performance Benchmarks
- **GPT-3 on WikiBio**: 71% AUC-PR for sentence-level detection
- **ChatGPT**: 74% AUC-PR
- Superior to grey-box methods requiring model access

### Implementation in Coherify
```python
from coherify import SelfCheckGPTAdapter, SelfCheckGPTEvaluator
adapter = SelfCheckGPTAdapter(method="bertscore")
evaluator = SelfCheckGPTEvaluator(coherence_measure)
```

### ⚠️ Implementation Notes
**Status**: ✅ **ENHANCED** - Core consistency algorithms implemented (Aug 2025)

**Previous Issue**: Had basic adapter but missing fundamental consistency checking algorithms
**Enhancement Added**: Complete implementation of BERTScore, NLI, N-gram, and QA-based consistency methods
**Performance Calibration**: SOTA detection models achieve 71-74% AUC-PR - coherence correlates with consistency

**Key Insight**: Focus on multi-response consistency detection rather than absolute accuracy

---

## FEVER (Fact Extraction and VERification)

### Paper Reference
**Title**: FEVER: a Large-scale Dataset for Fact Extraction and VERification  
**Authors**: James Thorne, Andreas Vlachos, Christos Christodoulopoulos, Arpit Mittal  
**Conference**: NAACL-HLT 2018  
**ArXiv**: [1803.05355](https://arxiv.org/abs/1803.05355)  
**Website**: [fever.ai](https://fever.ai/)  
**Code**: [GitHub](https://github.com/awslabs/fever)

### Dataset Characteristics
- **Total Claims**: 185,445 claims
- **Source**: Wikipedia articles (extracted and modified sentences)
- **Labels**: Supported, Refuted, NotEnoughInfo
- **Annotator Agreement**: κ = 0.6841 (Fleiss)

### Evidence Requirements
- **Single Sentence**: 68.25% of claims
- **Multiple Sentences**: 31.75% of claims  
- **Multiple Pages**: 12.15% of claims
- **Cross-page Evidence**: 16.82% of claims require evidence composition

### Task Definition
1. **Document Retrieval**: Find relevant Wikipedia pages
2. **Sentence Selection**: Identify evidence sentences
3. **Claim Verification**: Classify as Supported/Refuted/NotEnoughInfo

### Baseline Performance
- **With Correct Evidence**: 31.87% accuracy
- **Without Evidence Requirement**: 50.91% accuracy

### Implementation in Coherify
```python
from coherify import FEVERAdapter, EvidenceBasedCoherence
adapter = FEVERAdapter(include_evidence=True)
measure = EvidenceBasedCoherence()
```

### ⚠️ Implementation Notes  
**Status**: ✅ **ENHANCED** - Multi-sentence/multi-page evidence chains implemented (Aug 2025)

**Previous Issue**: Basic fact verification without proper evidence handling for complex cases
**Enhancement Added**: Evidence chain retrieval supporting 31.75% of claims requiring multiple sentences
**Performance Calibration**: Best published result 31.87% with evidence retrieval - challenging by design

**Key Statistics Addressed**:
- 31.75% claims requiring multiple sentences ✅
- 16.82% requiring evidence composition ✅  
- 12.15% requiring multiple Wikipedia pages ✅

---

## FaithBench

### Paper Reference
**Title**: FaithBench: A Diverse Hallucination Benchmark for Summarization by Modern LLMs  
**Authors**: Forrest Sheng Bao et al.  
**Date**: October 2024  
**ArXiv**: [2410.13210](https://arxiv.org/abs/2410.13210)  
**Papers with Code**: [Link](https://paperswithcode.com/paper/faithbench-a-diverse-hallucination-benchmark)

### Dataset Overview
- **Focus**: Summarization hallucination detection
- **Sources**: 10 modern LLMs from 8 different families
- **Annotation**: Human expert ground truth
- **Challenge Level**: Cases where SOTA detection models disagree

### Key Features
- **Diversity**: Wide range of topics and summarization styles
- **Recency**: Addresses limitations of existing hallucination benchmarks
- **Difficulty**: "Challenging" cases that trip up current detection methods
- **Real-world**: Based on actual LLM outputs, not synthetic data

### Performance Results
- **Best Generators**: GPT-4o and GPT-3.5-Turbo produce least hallucinations
- **Detection Challenge**: Even best detection models achieve ~50% accuracy
- **GPT-4o-as-Judge**: Struggles on challenging cases, highlighting room for improvement

### Hallucination Categories
- **Factual Errors**: Incorrect information not in source
- **Invented Details**: Made-up facts or statistics
- **Misrepresentation**: Distorted interpretation of source content
- **Context Violations**: Information contradicting source context

### Implementation in Coherify
```python
from coherify import FaithBenchAdapter, FaithfulnessCoherence
adapter = FaithBenchAdapter(challenge_level="hard")
measure = FaithfulnessCoherence()
```

### ⚠️ Implementation Notes
**Status**: ✅ **ENHANCED** - Challenging case filtering implemented (Aug 2025)

**Previous Issue**: Basic faithfulness checking without focusing on challenging cases where SOTA models disagree
**Enhancement Added**: Difficulty evaluation and filtering for cases with model disagreement
**Performance Calibration**: ~50% accuracy on challenging cases - designed to test edge cases where models fail

**Challenge Filtering**:
- **Easy**: <30% difficulty - Model agreement cases
- **Medium**: 30-70% difficulty - Some disagreement  
- **Hard**: >70% difficulty - High model disagreement

---

## Usage Guidelines

### Benchmark Selection Criteria
- **TruthfulQA**: Use for general truthfulness evaluation
- **SelfCheckGPT**: Use for hallucination detection without external resources
- **FEVER**: Use for fact-checking with evidence retrieval
- **FaithBench**: Use for summarization faithfulness evaluation

### ✅ Implementation Status Update (August 2025)

**RESOLVED**: All critical methodological issues have been fixed:

1. ✅ **TruthfulQA Evaluation Logic**: Fixed backward logic - now properly checks misinformation first
2. ✅ **SelfCheckGPT Core Algorithms**: Implemented complete consistency checking (BERTScore, NLI, N-gram, QA-based)
3. ✅ **FEVER Evidence Chains**: Added multi-sentence/multi-page evidence handling for complex claims
4. ✅ **FaithBench Challenge Focus**: Implemented challenging case filtering based on model disagreement
5. ✅ **Performance Expectations**: Calibrated realistic baselines based on published research

**Status**: All implementations complete with comprehensive test coverage. See `.claude/docs/benchmark_implementation_summary.md` for detailed results.

### Evaluation Best Practices
1. **Sample Size**: Start with small samples (10-50) for development
2. **Multiple Measures**: Compare different coherence approaches
3. **Baseline Comparison**: Always compare against native benchmark metrics
4. **Statistical Significance**: Use appropriate sample sizes for conclusions

### Performance Expectations
- **Processing Time**: Varies by measure complexity (0.1s - 2s per sample)
- **Memory Usage**: 200MB - 1GB depending on models loaded
- **API Costs**: $0.001 - 0.01 per sample for API-enhanced measures

### Common Pitfalls
- Confusing coherence scores with benchmark accuracy
- Using insufficient sample sizes for statistical validity
- Ignoring benchmark-specific evaluation protocols
- Over-interpreting results from single coherence measures

---

## References and Resources

### Additional Reading
- [Hallucination Survey Paper](https://arxiv.org/abs/2202.03629)
- [Coherence Theory Philosophy](https://plato.stanford.edu/entries/truth-coherence/)
- [NLP Evaluation Best Practices](https://aclanthology.org/2020.eval4nlp-1.9/)

### Community Resources
- **Papers with Code**: Benchmark leaderboards and implementations
- **Hugging Face**: Pre-trained models and datasets
- **OpenAI Research**: Latest evaluation methodologies

### Citation Guidelines
When using Coherify with these benchmarks, cite both the original benchmark papers and relevant coherence theory sources as appropriate for your research context.