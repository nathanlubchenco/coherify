# Coherify Current State - Quick Context

## 🎯 What This Project Does
Coherify uses **coherence measures to select better responses** from multiple LLM generations, improving performance on factuality benchmarks.

## ⚠️ Critical Understanding
**Coherence is NOT a replacement for benchmarks - it's an enhancement layer!**

The research pipeline has THREE stages:
1. **Baseline**: Generate 1 response → Evaluate 
2. **K-pass**: Generate K responses → Pick by majority vote → Evaluate
3. **Coherence**: Generate K responses → Pick by coherence → Evaluate

Goal: Show that Stage 3 > Stage 2 > Stage 1

## 🏗️ Current Architecture

```
/coherify/
├── generation/          # NEW: Actual model API calls
│   └── model_runner.py  # Handles OpenAI/Anthropic generation
├── evaluators/          # NEW: Response selection mechanisms  
│   └── response_selectors.py  # Majority voting & coherence selection
├── benchmarks/
│   ├── official/        # FIXED: Proper evaluation methods
│   │   ├── truthfulqa_official.py  # Uses GPT-4 judge, not fuzzy matching
│   │   └── truthfulqa_gpt4_judge.py  # NEW: GPT-4 evaluation
│   └── native_metrics.py  # Benchmark-specific metrics
├── measures/            # Coherence calculations (unchanged)
│   ├── semantic.py      # Embedding-based coherence
│   └── hybrid.py        # Combined approaches
└── examples/
    └── run_full_pipeline_comparison.py  # NEW: Compares all 3 stages
```

## 📊 What's Working Now

### ✅ FIXED Issues
1. **Model Generation**: Actually calls APIs (was using ground truth before!)
2. **Evaluation**: Uses GPT-4 as judge (was using string matching!)
3. **Pipeline**: All 3 stages implemented and comparable

### ⚠️ Current Issue
- Performance showing 0% - likely because:
  - No API key configured OR
  - Mock model returning empty/invalid responses
  - Need to validate evaluation logic with real responses

## 🚀 How to Run

### Quick Test (Mock)
```bash
make benchmark-full-pipeline MODEL=default SAMPLES=5 K_RUNS=3
```

### Real Test (Requires API Key)
```bash
export OPENAI_API_KEY=your-key-here
make benchmark-full-pipeline MODEL=gpt4-mini SAMPLES=20 K_RUNS=5
```

### Individual Stages
```bash
make benchmark-stage1 MODEL=gpt4-mini SAMPLES=20  # Baseline
make benchmark-stage2 MODEL=gpt4-mini SAMPLES=20 K_RUNS=5  # Majority voting
make benchmark-stage3 MODEL=gpt4-mini SAMPLES=20 K_RUNS=5  # Coherence selection
```

## 🔧 Configuration

**Models** (in `config/benchmark_config.json`):
- `default` → Mock model for testing
- `gpt4-mini` → OpenAI GPT-4o-mini
- `gpt4` → OpenAI GPT-4
- `claude` → Anthropic Claude

## 📈 Expected Results

With proper API configuration:
- **Stage 1 (Baseline)**: ~40-60% on TruthfulQA
- **Stage 2 (Majority)**: ~45-65% (5% improvement)
- **Stage 3 (Coherence)**: ~50-70% (5-10% over Stage 2)

## 🐛 Troubleshooting

**"Performance 0.0% is unrealistically low"**
- Check API keys are set
- Verify model is generating actual responses
- Check evaluation method (should use GPT-4 judge)

**"No module named X"**
- Run: `pip install -e ".[dev,benchmarks]"`

**"Failed to fetch" in UI**
- Reports may not be generated yet
- Check `results/` directory for JSON files

## 📝 Next Steps
1. Fix 0% performance issue (validate with real API)
2. Update all docs to reflect new pipeline
3. Test with multiple benchmarks
4. Validate improvement percentages

---
**Last Updated**: 2024-01-24
**Status**: Pipeline restructured, testing needed with real APIs