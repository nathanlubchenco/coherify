.PHONY: help install install-dev test test-verbose clean lint format type-check security build docs serve-docs benchmark coverage
.DEFAULT_GOAL := help

# Python environment
PYTHON := python
PIP := pip
PYTEST := pytest

# Project paths
SRC_DIR := coherify
TEST_DIR := tests
EXAMPLES_DIR := examples
DOCS_DIR := docs

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with development dependencies
	$(PIP) install -e ".[dev,benchmarks,viz,api,ui]"

test: ## Run tests
	$(PYTEST) $(TEST_DIR) -v

test-verbose: ## Run tests with verbose output and coverage
	$(PYTEST) $(TEST_DIR) -v --tb=long --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing

test-fast: ## Run tests without coverage
	$(PYTEST) $(TEST_DIR) -q

clean: ## Clean build artifacts and cache
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete

lint: ## Run linting checks
	@echo "Running flake8..."
	flake8 $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR) --count --show-source --statistics
	@echo "Running isort check..."
	isort --check-only $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

format: ## Format code with black and isort
	@echo "Formatting with black..."
	black $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "Sorting imports with isort..."
	isort $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

format-check: ## Check if code is properly formatted
	@echo "Checking black formatting..."
	black --check $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)
	@echo "Checking isort..."
	isort --check-only $(SRC_DIR) $(TEST_DIR) $(EXAMPLES_DIR)

type-check: ## Run type checking with mypy
	mypy $(SRC_DIR) --ignore-missing-imports

security: ## Run security checks
	@echo "Running bandit security linter..."
	bandit -r $(SRC_DIR)/ -f text
	@echo "Checking dependencies for known vulnerabilities..."
	safety check

build: ## Build package
	$(PYTHON) -m build

build-check: ## Build and check package
	$(PYTHON) -m build
	twine check dist/*

ci-quality: format-check lint type-check ## Run all quality checks (CI pipeline)

ci-test: test security ## Run all tests and security checks (CI pipeline)

ci-full: ci-quality ci-test build-check ## Run full CI pipeline locally

# Benchmark commands with configurable parameters
# Usage: make benchmark-truthfulqa MODEL=gpt4-mini SAMPLES=50
MODEL ?= default
SAMPLES ?= 100
K_RUNS ?= 1

benchmark-basic: ## Run basic benchmarks
	$(PYTHON) $(EXAMPLES_DIR)/basic_usage.py

benchmark-fever: ## Run FEVER benchmark (MODEL=model_name SAMPLES=N)
	$(PYTHON) $(EXAMPLES_DIR)/run_fever_benchmark.py --model $(MODEL) --sample-size $(SAMPLES)

benchmark-truthfulqa: ## Run TruthfulQA benchmark (MODEL=model_name SAMPLES=N K_RUNS=N)
	$(PYTHON) $(EXAMPLES_DIR)/run_truthfulqa_benchmark.py --model $(MODEL) --sample-size $(SAMPLES) --k-runs $(K_RUNS)

benchmark-faithbench: ## Run FaithBench benchmark (MODEL=model_name SAMPLES=N)
	$(PYTHON) $(EXAMPLES_DIR)/run_faithbench_benchmark.py --model $(MODEL) --sample-size $(SAMPLES)

benchmark-multi: ## Run multi-format benchmarks (MODEL=model_name SAMPLES=N)
	$(PYTHON) $(EXAMPLES_DIR)/run_multi_format_benchmarks.py --model $(MODEL) --sample-size $(SAMPLES)

benchmark-all: benchmark-fever benchmark-truthfulqa benchmark-faithbench benchmark-multi ## Run all benchmarks

# Stage-specific benchmarks for research pipeline
benchmark-stage1: ## Run Stage 1 (Official Baselines) - MODEL=model_name SAMPLES=N
	@echo "🏁 Running Stage 1: Official Baselines"
	$(PYTHON) $(EXAMPLES_DIR)/comprehensive_benchmark_demo.py --model $(MODEL) --sample-size $(SAMPLES) --stage 1

benchmark-stage2: ## Run Stage 2 (K-pass Majority Voting) - MODEL=model_name SAMPLES=N K_RUNS=N
	@echo "🔄 Running Stage 2: K-pass Majority Voting"
	$(PYTHON) $(EXAMPLES_DIR)/comprehensive_benchmark_demo.py --model $(MODEL) --sample-size $(SAMPLES) --k-runs $(K_RUNS) --stage 2

benchmark-stage3: ## Run Stage 3 (Coherence Enhancement) - MODEL=model_name SAMPLES=N K_RUNS=N
	@echo "🧠 Running Stage 3: Coherence-Enhanced Selection"
	$(PYTHON) $(EXAMPLES_DIR)/comprehensive_benchmark_demo.py --model $(MODEL) --sample-size $(SAMPLES) --k-runs $(K_RUNS) --stage 3

benchmark-full-pipeline: ## Run complete 3-stage research pipeline comparison
	@echo "🚀 Running Full Research Pipeline Comparison"
	$(PYTHON) $(EXAMPLES_DIR)/run_full_pipeline_comparison.py --model $(MODEL) --samples $(SAMPLES) --k-responses $(K_RUNS) --verbose

# Development commands
dev-setup: ## Set up development environment
	$(PIP) install --upgrade pip
	$(MAKE) install-dev
	@echo "Development environment set up successfully!"

dev-test: ## Run development test suite
	$(MAKE) format
	$(MAKE) ci-quality
	$(MAKE) test-verbose

coverage: ## Generate and open coverage report
	$(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html
	@echo "Coverage report generated in htmlcov/"
	@echo "Open htmlcov/index.html in your browser"

# Example commands
examples: ## Run all examples
	@echo "Running basic usage example..."
	$(PYTHON) $(EXAMPLES_DIR)/basic_usage.py
	@echo "Running phase 2 features example..."
	$(PYTHON) $(EXAMPLES_DIR)/phase2_features.py
	@echo "Running practical applications example..."
	$(PYTHON) $(EXAMPLES_DIR)/practical_applications.py

# Docker commands
docker-build: ## Build Docker image
	docker build -t coherify:dev .

docker-run: ## Run Docker container
	docker run -it --rm -v $(PWD):/app coherify:dev

docker-test: ## Run tests in Docker
	docker run --rm -v $(PWD):/app coherify:dev make test

# Release commands
tag-version: ## Tag current version (use VERSION=x.y.z)
	@if [ -z "$(VERSION)" ]; then echo "Please provide VERSION=x.y.z"; exit 1; fi
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

# Utility commands
outdated: ## Check for outdated dependencies
	$(PIP) list --outdated

update-deps: ## Update dependencies (be careful!)
	$(PIP) install --upgrade pip
	$(PIP) install --upgrade -e ".[dev,benchmarks,viz,api]"

requirements: ## Generate requirements.txt
	$(PIP) freeze > requirements.txt

.PHONY: jupyter ## Start Jupyter notebook
jupyter:
	jupyter notebook

.PHONY: shell ## Start Python shell with package imported
shell:
	$(PYTHON) -c "import coherify; print('Coherify imported successfully'); import IPython; IPython.start_ipython()"

# UI commands
ui: ## Launch the interactive Coherify UI
	$(PYTHON) run_ui.py

ui-dev: ## Launch UI in development mode with auto-reload
	streamlit run ui/coherence_app_v2.py --server.runOnSave true
