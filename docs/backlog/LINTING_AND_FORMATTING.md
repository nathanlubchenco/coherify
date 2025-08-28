# Linting and Formatting Setup

## Overview
The codebase now has comprehensive linting, formatting, and pre-commit hooks configured to maintain code quality and consistency.

## Tools Installed

### Code Formatting
- **Black** (v25.1.0): Python code formatter with 88-character line length
- **isort** (v6.0.1): Import statement sorting compatible with Black
- **autoflake** (v2.3.1): Removes unused imports and variables

### Code Linting
- **Flake8** (v7.3.0): Style guide enforcement
- **Bandit**: Security linting for common vulnerabilities
- **MyPy** (v1.17.1): Static type checking

### Pre-commit Framework
- **pre-commit** (v4.3.0): Git hook framework for automatic checks

## Configuration Files

### `.pre-commit-config.yaml`
Defines all hooks that run automatically before commits:
- Black formatting
- isort import sorting
- autoflake cleanup
- Flake8 linting
- Bandit security checks
- File cleanup (trailing whitespace, EOF fixes)
- JSON/YAML validation
- Type checking with MyPy

### `.flake8`
Configures Flake8 with:
- Line length: 88 (matches Black)
- Ignored rules for Black compatibility
- Per-file exceptions for examples and tests
- Max complexity: 15

### `pyproject.toml`
Contains tool configurations for:
- Black: line-length=88, target Python 3.10
- isort: Black-compatible profile
- MyPy: Permissive settings for gradual typing
- pytest: Test discovery settings

## Pre-commit Hooks Installed

### Stage 1: Formatting (runs automatically)
```bash
# These fix issues automatically:
- Black (Python formatting)
- isort (Import sorting)
- autoflake (Remove unused code)
- Trailing whitespace removal
- End-of-file fixes
- Line ending normalization (LF)
```

### Stage 2: Linting (reports issues)
```bash
# These check for issues:
- Flake8 (Style violations)
- Bandit (Security issues)
- Check for merge conflicts
- Validate JSON/YAML
- Debug statement detection
- Large file prevention (>1MB)
```

### Stage 3: Type Checking (optional)
```bash
# MyPy static type checking
- Currently permissive mode
- Excludes tests and examples
```

## Usage

### Run on All Files
```bash
# Check all files
pre-commit run --all-files

# Auto-fix what's possible
pre-commit run --all-files --hook-stage manual
```

### Run Specific Tools
```bash
# Format with Black
black coherify/ examples/ scripts/ tests/

# Sort imports
isort coherify/ examples/ scripts/ tests/

# Check linting
flake8 coherify/

# Type checking
mypy coherify/
```

### Manual Formatting Commands
```bash
# Format everything
make format

# Check formatting without changing
make format-check

# Run linting
make lint

# Type checking
make type-check
```

## Current State

### ✅ Completed
- All tools installed and configured
- Pre-commit hooks active on git commits
- Configuration files created
- Makefile commands available
- Basic formatting applied to all Python files

### 🔧 Improvements Made
1. **Consistent formatting**: All files use Black's style
2. **Import organization**: Sorted and grouped properly
3. **Unused code removed**: autoflake cleaned up imports
4. **Security scanning**: Bandit checks for vulnerabilities
5. **Automatic enforcement**: Pre-commit prevents bad commits

### ⚠️ Remaining Issues
Some minor issues remain that don't block functionality:
- A few unused variables in examples (intentional for demos)
- Some long lines in tests (test data)
- Complex functions that could be refactored

## Benefits

### For Development
- **Consistent style**: No debates about formatting
- **Automatic fixes**: Many issues fixed automatically
- **Early detection**: Catch issues before commit
- **Security awareness**: Bandit finds vulnerabilities

### For Collaboration
- **Clean history**: No formatting-only commits
- **Review focus**: PRs focus on logic, not style
- **CI/CD ready**: Same checks locally and in CI
- **Documentation**: Clear code standards

## Commit Workflow

When you commit, pre-commit will:
1. **Format** your code with Black
2. **Sort** imports with isort
3. **Remove** unused imports
4. **Check** for style issues
5. **Scan** for security problems

If issues are found:
```bash
# Pre-commit will show what failed
# Fix issues and re-commit
git add -u
git commit -m "Your message"
```

## Bypassing Hooks (Emergency Only)
```bash
# Skip pre-commit hooks (not recommended)
git commit --no-verify -m "Emergency fix"
```

## Updating Pre-commit
```bash
# Update hook versions
pre-commit autoupdate

# Clean and reinstall
pre-commit clean
pre-commit install
```

## CI/CD Integration

Add to your CI pipeline:
```yaml
# GitHub Actions example
- name: Run pre-commit
  uses: pre-commit/action@v3.0.0
```

## Summary

The codebase now has professional-grade linting and formatting that:
- ✅ Runs automatically on every commit
- ✅ Maintains consistent code style
- ✅ Catches bugs and security issues early
- ✅ Improves code review quality
- ✅ Reduces technical debt

All developers should have pre-commit installed and active!
