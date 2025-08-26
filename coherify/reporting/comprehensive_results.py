"""
Comprehensive benchmark result reporting and storage system.

Provides detailed reporting for benchmark evaluations including metrics,
context, examples, timing, cost estimation, and error analysis.
"""

import json
import time
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
import platform

# Import performance expectations for validation
try:
    from coherify.benchmarks.native_metrics import BenchmarkPerformanceExpectations
    HAS_PERFORMANCE_VALIDATION = True
except ImportError:
    HAS_PERFORMANCE_VALIDATION = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    HAS_TIKTOKEN = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class ModelInfo:
    """Information about the model used in evaluation."""
    name: Optional[str] = None
    provider: Optional[str] = None
    version: Optional[str] = None
    api_version: Optional[str] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    embedding_model: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkContext:
    """Published benchmark context and reference information."""
    benchmark_name: str
    version: Optional[str] = None
    paper_reference: Optional[str] = None
    homepage_url: Optional[str] = None
    dataset_size: Optional[int] = None
    published_results: Optional[Dict[str, float]] = None
    evaluation_metrics: Optional[List[str]] = None
    human_performance: Optional[float] = None
    state_of_art_performance: Optional[float] = None
    description: Optional[str] = None


@dataclass
class ExampleResult:
    """Individual example with input, output, and evaluation details."""
    input_text: str
    output_text: str
    expected_output: Optional[str] = None
    coherence_score: Optional[float] = None
    is_correct: Optional[bool] = None
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


@dataclass
class ErrorInfo:
    """Information about errors encountered during evaluation."""
    error_type: str
    error_message: str
    stack_trace: Optional[str] = None
    sample_index: Optional[int] = None
    timestamp: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkReport:
    """Comprehensive benchmark evaluation report."""
    
    # Basic Information
    benchmark_name: str
    evaluation_id: str
    timestamp: str
    duration_seconds: float
    
    # Model and System Info
    model_info: ModelInfo
    system_info: Dict[str, Any]
    
    # Benchmark Context
    benchmark_context: BenchmarkContext
    
    # Topline Metrics
    num_samples: int
    num_successful: int
    num_failed: int
    success_rate: float
    mean_coherence: float
    std_coherence: Optional[float]
    min_coherence: Optional[float]
    max_coherence: Optional[float]
    median_coherence: Optional[float]
    
    # Category Breakdown
    category_metrics: Dict[str, Dict[str, float]]
    
    # Native Benchmark Metrics
    native_metrics: Optional[Dict[str, Any]] = None
    benchmark_primary_metric: Optional[Tuple[str, float]] = None
    
    # Performance Metrics
    total_tokens_used: Optional[int] = None
    estimated_cost_usd: Optional[float] = None
    avg_time_per_sample: float = 0.0
    throughput_samples_per_second: float = 0.0
    
    # Examples
    correct_examples: List[ExampleResult] = None
    incorrect_examples: List[ExampleResult] = None
    edge_case_examples: List[ExampleResult] = None
    
    # Error Analysis
    errors: List[ErrorInfo] = None
    error_rate: float = 0.0
    error_categories: Dict[str, int] = None
    
    def __post_init__(self):
        """Initialize default values for lists."""
        if self.correct_examples is None:
            self.correct_examples = []
        if self.incorrect_examples is None:
            self.incorrect_examples = []
        if self.edge_case_examples is None:
            self.edge_case_examples = []
        if self.errors is None:
            self.errors = []
        if self.error_categories is None:
            self.error_categories = {}
        if self.evaluation_config is None:
            self.evaluation_config = {}
    
    # Additional Metrics
    coherence_distribution: Optional[Dict[str, int]] = None
    correlation_with_human: Optional[float] = None
    statistical_significance: Optional[Dict[str, Any]] = None
    
    # Configuration
    evaluation_config: Dict[str, Any] = None
    
    # Performance Validation
    performance_validation: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return asdict(self)
    
    def save_json(self, filepath: Union[str, Path]) -> None:
        """Save report as JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
    
    def save_markdown(self, filepath: Union[str, Path]) -> None:
        """Save report as Markdown file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        markdown = self._generate_markdown()
        with open(filepath, 'w') as f:
            f.write(markdown)
    
    def _generate_markdown(self) -> str:
        """Generate markdown report."""
        lines = []
        
        # Header
        lines.append(f"# {self.benchmark_name} Evaluation Report")
        lines.append(f"**Evaluation ID**: {self.evaluation_id}")
        lines.append(f"**Timestamp**: {self.timestamp}")
        lines.append(f"**Duration**: {self.duration_seconds:.2f}s")
        lines.append("")
        
        # Topline Metrics
        lines.append("## 📊 Topline Metrics")
        
        # Native benchmark metrics if available
        if self.native_metrics:
            lines.append("### Native Benchmark Performance")
            if self.benchmark_primary_metric:
                metric_name, metric_value = self.benchmark_primary_metric
                lines.append(f"- **Primary Metric ({metric_name})**: {metric_value:.3f}")
            
            # Show performance validation warnings if available
            if self.performance_validation:
                for metric_name, validation in self.performance_validation.items():
                    if not validation.get("is_realistic", True):
                        lines.append(f"⚠️  **Performance Warning ({metric_name})**: {validation.get('explanation', '')}")
                    elif "expectations" in validation:
                        exp = validation["expectations"]
                        if "best_model" in exp:
                            lines.append(f"ℹ️  **Research Context**: Best published {metric_name} {exp['best_model']:.1%}")
            
            if "truthful_score" in self.native_metrics:
                lines.append(f"- **Truthfulness**: {self.native_metrics['truthful_score']:.3f}")
                lines.append(f"- **Informativeness**: {self.native_metrics.get('informative_score', 0):.3f}")
            
            if "baseline_accuracy" in self.native_metrics:
                lines.append(f"- **Baseline Accuracy**: {self.native_metrics['baseline_accuracy']:.3f}")
                
            if "coherence_filtered_accuracy" in self.native_metrics and self.native_metrics['coherence_filtered_accuracy'] is not None:
                lines.append(f"- **Coherence-Filtered Accuracy**: {self.native_metrics['coherence_filtered_accuracy']:.3f}")
                if self.native_metrics.get('improvement') is not None:
                    improvement = self.native_metrics['improvement']
                    sign = "+" if improvement >= 0 else ""
                    lines.append(f"- **Improvement**: {sign}{improvement:.3f}")
                    
                    # Show expected improvement range if available
                    if self.performance_validation and "coherence_improvement" in self.performance_validation:
                        exp_range = self.performance_validation["coherence_improvement"].get("expected_range")
                        if exp_range and isinstance(exp_range, (list, tuple)) and len(exp_range) == 2:
                            lines.append(f"- **Expected Coherence Improvement**: {exp_range[0]:.1%}-{exp_range[1]:.1%}")
            lines.append("")
        
        lines.append("### Coherence Metrics")
        lines.append(f"- **Samples Evaluated**: {self.num_samples:,}")
        lines.append(f"- **Success Rate**: {self.success_rate:.1%}")
        lines.append(f"- **Mean Coherence**: {self.mean_coherence:.3f}")
        if self.std_coherence:
            lines.append(f"- **Standard Deviation**: {self.std_coherence:.3f}")
        if self.median_coherence:
            lines.append(f"- **Median Coherence**: {self.median_coherence:.3f}")
        if self.min_coherence is not None and self.max_coherence is not None:
            lines.append(f"- **Range**: {self.min_coherence:.3f} - {self.max_coherence:.3f}")
        lines.append("")
        
        # Model Information
        lines.append("## 🤖 Model Information")
        if self.model_info.name:
            lines.append(f"- **Model**: {self.model_info.name}")
        if self.model_info.provider:
            lines.append(f"- **Provider**: {self.model_info.provider}")
        if self.model_info.temperature:
            lines.append(f"- **Temperature**: {self.model_info.temperature}")
        if self.model_info.embedding_model:
            lines.append(f"- **Embedding Model**: {self.model_info.embedding_model}")
        lines.append("")
        
        # Performance Metrics
        lines.append("## ⚡ Performance Metrics")
        lines.append(f"- **Avg Time per Sample**: {self.avg_time_per_sample:.3f}s")
        lines.append(f"- **Throughput**: {self.throughput_samples_per_second:.2f} samples/sec")
        if self.total_tokens_used:
            lines.append(f"- **Total Tokens**: {self.total_tokens_used:,}")
        if self.estimated_cost_usd:
            lines.append(f"- **Estimated Cost**: ${self.estimated_cost_usd:.4f}")
        lines.append("")
        
        # Benchmark Context
        if self.benchmark_context.description:
            lines.append("## 📚 Benchmark Context")
            lines.append(f"**Description**: {self.benchmark_context.description}")
            if self.benchmark_context.paper_reference:
                lines.append(f"**Reference**: {self.benchmark_context.paper_reference}")
            if self.benchmark_context.human_performance:
                lines.append(f"**Human Performance**: {self.benchmark_context.human_performance:.3f}")
            if self.benchmark_context.state_of_art_performance:
                lines.append(f"**State-of-the-Art**: {self.benchmark_context.state_of_art_performance:.3f}")
            lines.append("")
        
        # Category Breakdown
        if self.category_metrics:
            lines.append("## 📂 Category Analysis")
            for category, metrics in self.category_metrics.items():
                lines.append(f"### {category}")
                if isinstance(metrics, dict):
                    for metric, value in metrics.items():
                        if isinstance(value, float):
                            lines.append(f"- **{metric}**: {value:.3f}")
                        else:
                            lines.append(f"- **{metric}**: {value}")
                else:
                    # Handle case where metrics is a direct float value
                    lines.append(f"- **Mean Coherence**: {metrics:.3f}")
                lines.append("")
        
        # Examples
        if self.correct_examples:
            lines.append("## ✅ Correct Examples")
            for i, example in enumerate(self.correct_examples[:3]):  # Show top 3
                lines.append(f"### Example {i+1}")
                lines.append(f"**Input**: {example.input_text[:200]}...")
                lines.append(f"**Output**: {example.output_text[:200]}...")
                lines.append(f"**Coherence**: {example.coherence_score:.3f}")
                lines.append("")
        
        if self.incorrect_examples:
            lines.append("## ❌ Incorrect Examples")
            for i, example in enumerate(self.incorrect_examples[:3]):  # Show top 3
                lines.append(f"### Example {i+1}")
                lines.append(f"**Input**: {example.input_text[:200]}...")
                lines.append(f"**Output**: {example.output_text[:200]}...")
                lines.append(f"**Coherence**: {example.coherence_score:.3f}")
                if example.error_message:
                    lines.append(f"**Error**: {example.error_message}")
                lines.append("")
        
        # Error Analysis
        if self.errors:
            lines.append("## 🚨 Error Analysis")
            lines.append(f"- **Error Rate**: {self.error_rate:.1%}")
            lines.append("- **Error Categories**:")
            for error_type, count in self.error_categories.items():
                lines.append(f"  - {error_type}: {count}")
            lines.append("")
        
        # System Information
        lines.append("## 💻 System Information")
        for key, value in self.system_info.items():
            lines.append(f"- **{key}**: {value}")
        
        return "\n".join(lines)


class BenchmarkReporter:
    """Comprehensive benchmark reporter that creates detailed evaluation reports."""
    
    def __init__(self, results_dir: Union[str, Path] = "results"):
        """Initialize benchmark reporter."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Benchmark context database
        self._benchmark_contexts = self._load_benchmark_contexts()
    
    def create_report(
        self,
        benchmark_name: str,
        raw_results: Dict[str, Any],
        model_info: Optional[ModelInfo] = None,
        evaluation_config: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        examples: Optional[List[ExampleResult]] = None,
        errors: Optional[List[ErrorInfo]] = None,
    ) -> BenchmarkReport:
        """Create comprehensive benchmark report from raw evaluation results."""
        
        # Generate evaluation ID
        eval_id = self._generate_evaluation_id(benchmark_name)
        
        # Calculate timing
        if start_time and end_time:
            duration = end_time - start_time
        else:
            duration = raw_results.get('eval_time', 0)
        
        # Extract basic metrics
        num_samples = raw_results.get('num_samples', 0)
        num_failed = len(errors) if errors else 0
        num_successful = num_samples - num_failed
        success_rate = num_successful / num_samples if num_samples > 0 else 0
        
        # Calculate coherence statistics
        coherence_stats = self._calculate_coherence_statistics(raw_results)
        
        # Get benchmark context
        benchmark_context = self._get_benchmark_context(benchmark_name)
        
        # Extract examples
        correct_examples, incorrect_examples, edge_cases = self._extract_examples(
            raw_results, examples or []
        )
        
        # Calculate performance metrics
        perf_metrics = self._calculate_performance_metrics(
            raw_results, duration, num_samples, model_info
        )
        
        # Analyze errors
        error_analysis = self._analyze_errors(errors or [])
        
        # Create report
        report = BenchmarkReport(
            benchmark_name=benchmark_name,
            evaluation_id=eval_id,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
            
            model_info=model_info or ModelInfo(),
            system_info=self._get_system_info(),
            benchmark_context=benchmark_context,
            
            num_samples=num_samples,
            num_successful=num_successful,
            num_failed=num_failed,
            success_rate=success_rate,
            
            mean_coherence=coherence_stats['mean'],
            std_coherence=coherence_stats['std'],
            min_coherence=coherence_stats['min'],
            max_coherence=coherence_stats['max'],
            median_coherence=coherence_stats['median'],
            
            native_metrics=raw_results.get('native_metrics'),
            benchmark_primary_metric=raw_results.get('benchmark_primary_metric'),
            performance_validation=self._validate_performance(benchmark_name, raw_results.get('native_metrics')),
            
            category_metrics=raw_results.get('category_means', {}),
            
            total_tokens_used=perf_metrics['tokens'],
            estimated_cost_usd=perf_metrics['cost'],
            avg_time_per_sample=duration / num_samples if num_samples > 0 else 0,
            throughput_samples_per_second=num_samples / duration if duration > 0 else 0,
            
            correct_examples=correct_examples,
            incorrect_examples=incorrect_examples,
            edge_case_examples=edge_cases,
            
            errors=errors or [],
            error_rate=num_failed / num_samples if num_samples > 0 else 0,
            error_categories=error_analysis['categories'],
            
            coherence_distribution=self._calculate_distribution(raw_results),
            correlation_with_human=None,  # TODO: Implement if human annotations available
            statistical_significance=None,  # TODO: Implement statistical tests
            
            evaluation_config=evaluation_config or {},
        )
        
        return report
    
    def save_report(self, report: BenchmarkReport) -> Tuple[Path, Path]:
        """Save report as both JSON and Markdown files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_filename = f"{report.benchmark_name}_{timestamp}_{report.evaluation_id[:8]}"
        
        json_path = self.results_dir / f"{base_filename}.json"
        md_path = self.results_dir / f"{base_filename}.md"
        
        report.save_json(json_path)
        report.save_markdown(md_path)
        
        return json_path, md_path
    
    def list_reports(self, benchmark_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available reports with metadata."""
        reports = []
        
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                if benchmark_name and data.get('benchmark_name') != benchmark_name:
                    continue
                
                reports.append({
                    'filename': json_file.name,
                    'benchmark_name': data.get('benchmark_name'),
                    'timestamp': data.get('timestamp'),
                    'num_samples': data.get('num_samples'),
                    'mean_coherence': data.get('mean_coherence'),
                    'success_rate': data.get('success_rate'),
                })
            except Exception:
                continue
        
        return sorted(reports, key=lambda x: x['timestamp'], reverse=True)
    
    def _generate_evaluation_id(self, benchmark_name: str) -> str:
        """Generate unique evaluation ID."""
        content = f"{benchmark_name}_{time.time()}_{platform.node()}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _calculate_coherence_statistics(self, raw_results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate comprehensive coherence statistics."""
        stats = {
            'mean': raw_results.get('mean_coherence', 0),
            'std': None,
            'min': None,
            'max': None,
            'median': None,
        }
        
        # Extract all coherence scores if available
        all_scores = []
        if 'detailed_results' in raw_results:
            all_scores = [r.get('coherence_score', 0) for r in raw_results['detailed_results']]
        
        if all_scores and HAS_NUMPY:
            scores_array = np.array(all_scores)
            stats.update({
                'std': float(np.std(scores_array)),
                'min': float(np.min(scores_array)),
                'max': float(np.max(scores_array)),
                'median': float(np.median(scores_array)),
            })
        elif all_scores:
            # Fallback without numpy
            stats.update({
                'min': min(all_scores),
                'max': max(all_scores),
                'median': sorted(all_scores)[len(all_scores) // 2],
            })
            
            # Simple std calculation
            mean = sum(all_scores) / len(all_scores)
            variance = sum((x - mean) ** 2 for x in all_scores) / len(all_scores)
            stats['std'] = variance ** 0.5
        
        return stats
    
    def _extract_examples(
        self, raw_results: Dict[str, Any], examples: List[ExampleResult]
    ) -> Tuple[List[ExampleResult], List[ExampleResult], List[ExampleResult]]:
        """Extract correct, incorrect, and edge case examples."""
        
        if examples:
            # Use provided examples
            correct = [ex for ex in examples if ex.is_correct]
            incorrect = [ex for ex in examples if ex.is_correct is False]
            edge_cases = [ex for ex in examples if ex.is_correct is None]
        else:
            # Extract from detailed results
            correct = []
            incorrect = []
            edge_cases = []
            
            detailed = raw_results.get('detailed_results', [])
            for result in detailed[:10]:  # Limit to prevent huge files
                coherence = result.get('coherence_score', 0)
                
                example = ExampleResult(
                    input_text=str(result.get('input', 'N/A')),
                    output_text=str(result.get('output', 'N/A')),
                    coherence_score=coherence,
                    category=result.get('category'),
                )
                
                # Classify as correct/incorrect based on coherence threshold
                if coherence > 0.7:
                    correct.append(example)
                elif coherence < 0.3:
                    incorrect.append(example)
                else:
                    edge_cases.append(example)
        
        # Sort by coherence score
        correct.sort(key=lambda x: x.coherence_score or 0, reverse=True)
        incorrect.sort(key=lambda x: x.coherence_score or 0)
        
        return correct[:5], incorrect[:5], edge_cases[:3]
    
    def _calculate_performance_metrics(
        self, raw_results: Dict[str, Any], duration: float, num_samples: int, model_info: Optional[ModelInfo]
    ) -> Dict[str, Optional[Union[int, float]]]:
        """Calculate performance metrics including cost estimation."""
        
        tokens = None
        cost = None
        
        # Try to extract token usage
        if 'api_statistics' in raw_results:
            tokens = raw_results['api_statistics'].get('total_tokens')
        
        # Estimate cost based on model and tokens
        if tokens and model_info and model_info.provider:
            cost = self._estimate_cost(model_info.provider, model_info.name, tokens)
        
        return {
            'tokens': tokens,
            'cost': cost,
        }
    
    def _estimate_cost(self, provider: str, model: str, tokens: int) -> float:
        """Estimate API cost based on provider, model, and token usage."""
        # Cost per 1K tokens (as of Jan 2025)
        pricing = {
            'openai': {
                'gpt-4o': {'input': 0.0025, 'output': 0.01},
                'gpt-4o-mini': {'input': 0.00015, 'output': 0.0006},
                'text-embedding-3-small': {'input': 0.00002, 'output': 0},
            },
            'anthropic': {
                'claude-3-opus': {'input': 0.015, 'output': 0.075},
                'claude-3-sonnet': {'input': 0.003, 'output': 0.015},
                'claude-3-haiku': {'input': 0.00025, 'output': 0.00125},
            }
        }
        
        if provider in pricing and model in pricing[provider]:
            # Assume equal input/output split
            rates = pricing[provider][model]
            avg_rate = (rates['input'] + rates['output']) / 2
            return (tokens / 1000) * avg_rate
        
        return None
    
    def _analyze_errors(self, errors: List[ErrorInfo]) -> Dict[str, Any]:
        """Analyze error patterns."""
        categories = {}
        for error in errors:
            error_type = error.error_type
            categories[error_type] = categories.get(error_type, 0) + 1
        
        return {
            'categories': categories,
        }
    
    def _calculate_distribution(self, raw_results: Dict[str, Any]) -> Optional[Dict[str, int]]:
        """Calculate coherence score distribution."""
        detailed = raw_results.get('detailed_results', [])
        if not detailed:
            return None
        
        scores = [r.get('coherence_score', 0) for r in detailed]
        bins = {
            '0.0-0.2': 0,
            '0.2-0.4': 0,
            '0.4-0.6': 0,
            '0.6-0.8': 0,
            '0.8-1.0': 0,
        }
        
        for score in scores:
            if score < 0.2:
                bins['0.0-0.2'] += 1
            elif score < 0.4:
                bins['0.2-0.4'] += 1
            elif score < 0.6:
                bins['0.4-0.6'] += 1
            elif score < 0.8:
                bins['0.6-0.8'] += 1
            else:
                bins['0.8-1.0'] += 1
        
        return bins
    
    def _get_system_info(self) -> Dict[str, Any]:
        """Get system information."""
        info = {
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
        }
        
        if HAS_PSUTIL:
            info.update({
                'cpu_count': psutil.cpu_count(),
                'memory_gb': round(psutil.virtual_memory().total / (1024**3), 2),
            })
        else:
            import os
            info.update({
                'cpu_count': os.cpu_count() or 'Unknown',
                'memory_gb': 'Unknown',
            })
        
        return info
    
    def _load_benchmark_contexts(self) -> Dict[str, BenchmarkContext]:
        """Load benchmark context information."""
        return {
            'truthfulqa': BenchmarkContext(
                benchmark_name='TruthfulQA',
                version='1.0',
                paper_reference='Lin et al. (2021). TruthfulQA: Measuring How Models Mimic Human Falsehoods',
                homepage_url='https://github.com/sylinrl/TruthfulQA',
                dataset_size=817,
                published_results={
                    'GPT-3 (175B)': 0.58,
                    'GPT-2': 0.40,
                    'UnifiedQA': 0.68,
                },
                evaluation_metrics=['MC1', 'MC2', 'Generation'],
                human_performance=0.94,
                state_of_art_performance=0.75,
                description='TruthfulQA measures whether a language model is truthful in generating answers to questions by testing on questions spanning 38 categories including health, law, finance, and politics.',
            ),
            'selfcheckgpt': BenchmarkContext(
                benchmark_name='SelfCheckGPT',
                version='1.0',
                paper_reference='Manakul et al. (2023). SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection',
                homepage_url='https://github.com/potsawee/selfcheckgpt',
                published_results={
                    'GPT-3': 0.71,
                    'ChatGPT': 0.74,
                },
                evaluation_metrics=['AUROC', 'Precision', 'Recall'],
                description='SelfCheckGPT is a method for hallucination detection that uses the principle of sampling multiple responses and checking consistency.',
            ),
        }
    
    def _get_benchmark_context(self, benchmark_name: str) -> BenchmarkContext:
        """Get benchmark context or create a default one."""
        return self._benchmark_contexts.get(
            benchmark_name.lower(),
            BenchmarkContext(benchmark_name=benchmark_name)
        )
    
    def _validate_performance(self, benchmark_name: str, native_metrics: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Validate performance metrics against research baselines."""
        if not HAS_PERFORMANCE_VALIDATION or not native_metrics:
            return None
        
        validation_results = {}
        
        # Get benchmark expectations
        expectations = BenchmarkPerformanceExpectations.get_expectations(benchmark_name.upper())
        if not expectations:
            return None
        
        # Validate key metrics based on benchmark type
        if benchmark_name.lower() in ["truthfulqa", "truthful_qa"]:
            if "truthful_score" in native_metrics:
                is_realistic, explanation = BenchmarkPerformanceExpectations.is_performance_realistic(
                    "truthfulqa", native_metrics["truthful_score"]
                )
                validation_results["truthfulness"] = {
                    "is_realistic": is_realistic,
                    "explanation": explanation,
                    "expectations": expectations
                }
            
            # Add coherence improvement expectations
            if "improvement" in native_metrics:
                improvement_range = expectations.get("coherence_improvement", (0, 0))
                validation_results["coherence_improvement"] = {
                    "expected_range": improvement_range,
                    "actual": native_metrics["improvement"]
                }
        
        elif benchmark_name.lower() in ["selfcheckgpt", "self_check_gpt"]:
            # Look for consistency-related metrics
            for key in ["accuracy", "auc_pr", "consistency_score"]:
                if key in native_metrics:
                    is_realistic, explanation = BenchmarkPerformanceExpectations.is_performance_realistic(
                        "selfcheckgpt", native_metrics[key]
                    )
                    validation_results[key] = {
                        "is_realistic": is_realistic,
                        "explanation": explanation,
                        "expectations": expectations
                    }
        
        elif benchmark_name.lower() == "fever":
            if "baseline_accuracy" in native_metrics:
                is_realistic, explanation = BenchmarkPerformanceExpectations.is_performance_realistic(
                    "fever", native_metrics["baseline_accuracy"]
                )
                validation_results["accuracy"] = {
                    "is_realistic": is_realistic,
                    "explanation": explanation,
                    "expectations": expectations
                }
        
        elif benchmark_name.lower() in ["faithbench", "faith_bench"]:
            if "baseline_accuracy" in native_metrics:
                is_realistic, explanation = BenchmarkPerformanceExpectations.is_performance_realistic(
                    "faithbench", native_metrics["baseline_accuracy"]
                )
                validation_results["accuracy"] = {
                    "is_realistic": is_realistic,
                    "explanation": explanation,
                    "expectations": expectations
                }
        
        return validation_results if validation_results else None