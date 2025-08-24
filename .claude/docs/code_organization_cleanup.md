# Code Organization Cleanup - August 23, 2025

## Summary of Changes

The project structure has been cleaned up to improve code organization and prevent future clutter from temporary development files.

## Files Moved

### Validation Tests → Formal Tests Directory
The following validation test files were moved from the root directory to `tests/`:

- `test_truthfulqa_fix.py` → `tests/test_truthfulqa_validation.py`
- `test_selfcheckgpt_fix.py` → `tests/test_selfcheckgpt_validation.py` 
- `test_fever_evidence_chains.py` → `tests/test_fever_validation.py`
- `test_faithbench_challenging_cases.py` → `tests/test_faithbench_validation.py`
- `test_performance_expectations.py` → `tests/test_performance_validation.py`

**Rationale**: These are comprehensive validation tests for the benchmark implementation fixes and belong in the formal test structure rather than cluttering the root directory.

## New Structure Added

### Temporary Scripts Directory
Created `.tmp/` directory with the following structure:

```
.tmp/
├── README.md                    # Guidelines for temporary scripts
└── (temporary files here)       # Git-ignored development files
```

**Purpose**: Provides a designated space for temporary, experimental, and one-off test scripts during development.

## Git Configuration Updated

### .gitignore Additions
Added the following entries to `.gitignore`:

```gitignore
# Temporary development and testing files
.tmp/
*.tmp
*_temp.py
*_scratch.py
```

**Effect**: All temporary development files are now automatically ignored by git.

## Documentation Updates

### CLAUDE.md Enhancements
Added comprehensive guidelines for code organization:

#### Temporary Scripts Guidelines
- Use `.tmp/` for quick tests, debugging, prototypes, scratch code
- Automatic cleanup via git-ignore
- No formal structure required
- Clear file naming conventions

#### Formal Testing Guidelines  
- Use `tests/` for permanent, structured tests
- Follow proper conventions and documentation
- Git-tracked and part of CI/CD pipeline
- Run with `pytest tests/`

## Benefits

✅ **Clean Root Directory**: No more temporary test files cluttering the main codebase  
✅ **Clear Separation**: Formal tests vs temporary experiments have designated spaces  
✅ **Automatic Cleanup**: Git ignores temporary files, preventing accidental commits  
✅ **Developer Guidance**: Clear instructions on where different types of code belong  
✅ **Preserved Validation**: Important validation tests moved to proper formal test structure

## Usage Examples

### For Temporary Development
```bash
# Quick feature test
.tmp/test_new_coherence_idea.py

# Debug specific issue  
.tmp/debug_benchmark_problem.py

# Experimental prototype
.tmp/scratch_advanced_filtering.py
```

### For Formal Testing
```bash
# Run all formal tests
pytest tests/

# Run specific validation test
pytest tests/test_truthfulqa_validation.py

# Run specific test function
pytest tests/test_measures.py::test_semantic_coherence
```

This organization maintains a clean, professional codebase while providing developers the flexibility they need for experimental work.