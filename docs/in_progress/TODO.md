# Coherify Project TODO Tracker

## 🎯 Current Focus
**Stage**: Post-restructuring cleanup and documentation
**Priority**: Ensure all docs reflect the correct 3-stage pipeline approach

## ✅ Recently Completed (Critical Fixes)
- [x] Connected real model APIs for actual generation (not mock data)
- [x] Fixed evaluators to use GPT-4 judge instead of fuzzy matching
- [x] Implemented K-pass generation system
- [x] Added majority voting selector (Stage 2)
- [x] Added coherence-based selector (Stage 3)
- [x] Created full pipeline comparison framework
- [x] Fixed runtime errors in model providers

## 📋 Immediate Tasks (In Progress)
- [x] Implement unified benchmark runner (`coherify.benchmark_runner`)
- [x] Reorganize docs (in_progress/completed/backlog structure)
- [x] Add validation tests for baselines
- [ ] Fix FEVER evaluator for unified runner
- [ ] Test full pipeline with real API keys
- [ ] Migrate all example scripts to use unified runner

## 🎯 Unified Benchmark Runner ✅
- **Status**: COMPLETE
- **Location**: `coherify/benchmark_runner.py`
- **Features**:
  - Single entry point for all benchmarks
  - Configuration-driven execution
  - Automatic 3-stage pipeline
  - Results saving and comparison
- **Usage**: `python -m coherify.benchmark_runner truthfulqa --model gpt4-mini`

## 🔄 3-Stage Pipeline Implementation Status

### Stage 1: Official Baselines ✅
- **Status**: COMPLETE
- **What it does**: Single response → Official evaluation → Baseline score
- **Key files**:
  - `/coherify/benchmarks/official/truthfulqa_official.py`
  - `/coherify/generation/model_runner.py`

### Stage 2: K-pass Majority Voting ✅
- **Status**: COMPLETE
- **What it does**: K responses → Majority vote → Official evaluation → K-pass score
- **Key files**:
  - `/coherify/evaluators/response_selectors.py` (MajorityVotingSelector)
  - `/coherify/generation/model_runner.py` (KPassGenerator)

### Stage 3: Coherence Selection ✅
- **Status**: COMPLETE
- **What it does**: K responses → Coherence select → Official evaluation → Enhanced score
- **Key files**:
  - `/coherify/evaluators/response_selectors.py` (CoherenceSelector)
  - All coherence measures in `/coherify/measures/`

## 📝 Documentation Updates Needed
1. **CLAUDE.md**: Update implementation strategy to reflect 3-stage pipeline
2. **README.md**: Fix project description to emphasize coherence as enhancement
3. **benchmark_compatibility.md**: Rename/rewrite as pipeline methodology doc
4. **agent_context/**: Clean up outdated task files
5. **Code comments**: Remove TODOs that are now complete

## 🧪 Testing & Validation Required
1. **With Real Models**:
   - [ ] Test with GPT-4o-mini using actual OpenAI API
   - [ ] Test with Claude using Anthropic API
   - [ ] Compare results to published baselines

2. **Benchmark Coverage**:
   - [ ] TruthfulQA - Full pipeline test
   - [ ] FEVER - Implement 3-stage pipeline
   - [ ] SelfCheckGPT - Implement 3-stage pipeline

3. **Statistical Validation**:
   - [ ] Run with N=100+ samples for significance
   - [ ] Calculate confidence intervals
   - [ ] Validate coherence advantage is consistent

## 🚀 Future Enhancements (Post-Validation)
1. **Optimization**:
   - [ ] Batch API calls for efficiency
   - [ ] Cache embeddings and coherence scores
   - [ ] Parallel processing for K-pass generation

2. **Extended Evaluation**:
   - [ ] Test different K values (3, 5, 10, 20)
   - [ ] Try different coherence measures
   - [ ] Temperature variation strategies

3. **New Benchmarks**:
   - [ ] Add HaluEval support
   - [ ] Add FaithDial support
   - [ ] Create custom factuality benchmark

## 🐛 Known Issues
1. **API Integration**:
   - OpenAI provider needs response_format support
   - Anthropic provider not fully tested
   - Mock provider should simulate more realistic responses

2. **Evaluation Methods**:
   - GPT-4 judge prompts may need tuning
   - BLEURT installation not automated
   - Embedding similarity is too lenient

3. **UI/Reporting**:
   - Pipeline comparison not shown in UI
   - Stage comparisons need better visualization
   - Cost tracking not implemented

## 📊 Success Metrics
The project will be considered successful when:
1. ✅ All three stages implemented and working
2. ⬜ Coherence selection shows 5-10% improvement over majority voting
3. ⬜ Results validated against published baselines
4. ⬜ Full documentation updated and accurate
5. ⬜ Reproducible with simple make commands

## 🔗 Quick Commands
```bash
# Test full pipeline
make benchmark-full-pipeline MODEL=gpt4-mini SAMPLES=50 K_RUNS=5

# Run specific stage
make benchmark-stage1 MODEL=gpt4-mini SAMPLES=50
make benchmark-stage2 MODEL=gpt4-mini SAMPLES=50 K_RUNS=5
make benchmark-stage3 MODEL=gpt4-mini SAMPLES=50 K_RUNS=5

# Start UI
python examples/comprehensive_benchmark_demo.py --ui-only
```

## 📅 Last Updated
2024-01-24 - Post major restructuring to fix research methodology

---
**Note**: This TODO list is the source of truth for project status. Update it whenever tasks are completed or new issues are discovered.
