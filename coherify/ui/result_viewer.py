"""
Simple web-based result viewer for benchmark reports.

Provides a lightweight HTTP server to view comprehensive benchmark results.
"""

import json
import os
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Optional
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import threading
import time


class ResultViewerHandler(BaseHTTPRequestHandler):
    """HTTP request handler for serving benchmark results."""
    
    # Class variable to store results directory
    results_dir = None
    
    def log_message(self, format, *args):
        """Override to reduce logging noise."""
        pass
    
    def do_GET(self):
        """Handle GET requests."""
        path = urlparse(self.path).path
        query = parse_qs(urlparse(self.path).query)
        
        try:
            if path == "/":
                self._serve_index()
            elif path == "/api/reports":
                self._serve_reports_list()
            elif path == "/api/report":
                report_file = query.get('file', [None])[0]
                if report_file:
                    self._serve_report(report_file)
                else:
                    self._send_error(400, "Missing file parameter")
            elif path.startswith("/static/"):
                self._serve_static(path[8:])  # Remove /static/ prefix
            else:
                self._send_error(404, "Not found")
        except Exception as e:
            self._send_error(500, str(e))
    
    def _serve_index(self):
        """Serve the main index page."""
        html = self._generate_index_html()
        self._send_html(html)
    
    def _serve_reports_list(self):
        """Serve list of available reports as JSON."""
        reports = self._get_reports_list()
        self._send_json(reports)
    
    def _serve_report(self, filename: str):
        """Serve a specific report as JSON."""
        try:
            report_path = self.results_dir / filename
            if not report_path.exists():
                self._send_error(404, f"Report not found: {filename}")
                return
            
            with open(report_path) as f:
                report_data = json.load(f)
            
            self._send_json(report_data)
        except Exception as e:
            self._send_error(500, f"Error loading report: {e}")
    
    def _serve_static(self, filename: str):
        """Serve static files (CSS, JS)."""
        # For simplicity, we'll inline CSS/JS in the HTML
        self._send_error(404, "Static file not found")
    
    def _get_reports_list(self) -> List[Dict[str, Any]]:
        """Get list of available reports with metadata."""
        reports = []
        
        for json_file in self.results_dir.glob("*.json"):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                
                # Extract key metadata for the list view
                reports.append({
                    'filename': json_file.name,
                    'benchmark_name': data.get('benchmark_name', 'Unknown'),
                    'timestamp': data.get('timestamp', ''),
                    'num_samples': data.get('num_samples', 0),
                    'mean_coherence': data.get('mean_coherence', 0),
                    'success_rate': data.get('success_rate', 0),
                    'duration_seconds': data.get('duration_seconds', 0),
                    'model_name': data.get('model_info', {}).get('name', 'Unknown'),
                })
            except Exception:
                # Skip invalid files
                continue
        
        # Sort by timestamp, newest first
        reports.sort(key=lambda x: x['timestamp'], reverse=True)
        return reports
    
    def _generate_index_html(self) -> str:
        """Generate the main HTML page."""
        return '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Coherify - Benchmark Results</title>
    <style>
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif; 
            margin: 0; 
            padding: 20px; 
            background: #f5f5f5; 
        }
        .container { max-width: 1200px; margin: 0 auto; }
        .header { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .header h1 { margin: 0; color: #2c3e50; }
        .header p { margin: 5px 0 0 0; color: #7f8c8d; }
        .reports-grid { display: grid; grid-template-columns: 1fr; gap: 15px; }
        .report-card { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .report-card:hover { 
            transform: translateY(-2px); 
            box-shadow: 0 4px 8px rgba(0,0,0,0.15); 
        }
        .report-header { display: flex; justify-content: between; align-items: center; margin-bottom: 15px; }
        .report-title { font-size: 18px; font-weight: bold; color: #2c3e50; margin: 0; }
        .report-timestamp { font-size: 14px; color: #7f8c8d; }
        .report-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 15px; }
        .stat { text-align: center; }
        .stat-value { font-size: 24px; font-weight: bold; color: #3498db; }
        .stat-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
        .loading { text-align: center; padding: 40px; color: #7f8c8d; }
        .detail-view { 
            background: white; 
            padding: 20px; 
            border-radius: 8px; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: none;
        }
        .back-btn { 
            background: #3498db; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 4px; 
            cursor: pointer; 
            margin-bottom: 20px;
        }
        .back-btn:hover { background: #2980b9; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin: 20px 0; }
        .metric-card { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 6px; 
            text-align: center; 
        }
        .metric-value { font-size: 20px; font-weight: bold; color: #2c3e50; }
        .metric-label { font-size: 14px; color: #6c757d; margin-top: 5px; }
        .examples-section { margin-top: 30px; }
        .example-card { 
            background: #f8f9fa; 
            padding: 15px; 
            margin: 10px 0; 
            border-radius: 6px; 
            border-left: 4px solid #3498db;
        }
        .example-input { margin-bottom: 10px; }
        .example-output { margin-bottom: 10px; }
        .example-score { font-weight: bold; color: #27ae60; }
        .error { color: #e74c3c; background: #fdf2f2; padding: 15px; border-radius: 6px; margin: 10px 0; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📊 Coherify Benchmark Results</h1>
            <p>Comprehensive evaluation reports for coherence measures</p>
        </div>
        
        <div id="reports-list" class="reports-grid">
            <div class="loading">Loading reports...</div>
        </div>
        
        <div id="detail-view" class="detail-view">
            <button class="back-btn" onclick="showReportsList()">← Back to Reports</button>
            <div id="detail-content"></div>
        </div>
    </div>

    <script>
        let currentReports = [];
        
        async function loadReports() {
            try {
                const response = await fetch('/api/reports');
                currentReports = await response.json();
                displayReportsList();
            } catch (error) {
                document.getElementById('reports-list').innerHTML = 
                    '<div class="error">Error loading reports: ' + error.message + '</div>';
            }
        }
        
        function displayReportsList() {
            const container = document.getElementById('reports-list');
            
            if (currentReports.length === 0) {
                container.innerHTML = '<div class="loading">No reports found. Run a benchmark evaluation to generate reports.</div>';
                return;
            }
            
            container.innerHTML = currentReports.map(report => `
                <div class="report-card" onclick="viewReport('${report.filename}')">
                    <div class="report-header">
                        <h3 class="report-title">${report.benchmark_name}</h3>
                        <span class="report-timestamp">${formatTimestamp(report.timestamp)}</span>
                    </div>
                    <div class="report-stats">
                        <div class="stat">
                            <div class="stat-value">${report.num_samples.toLocaleString()}</div>
                            <div class="stat-label">Samples</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${report.mean_coherence.toFixed(3)}</div>
                            <div class="stat-label">Mean Coherence</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${(report.success_rate * 100).toFixed(1)}%</div>
                            <div class="stat-label">Success Rate</div>
                        </div>
                        <div class="stat">
                            <div class="stat-value">${report.duration_seconds.toFixed(1)}s</div>
                            <div class="stat-label">Duration</div>
                        </div>
                    </div>
                    <div style="margin-top: 10px; font-size: 14px; color: #6c757d;">
                        Model: ${report.model_name}
                    </div>
                </div>
            `).join('');
        }
        
        async function viewReport(filename) {
            try {
                const response = await fetch('/api/report?file=' + encodeURIComponent(filename));
                const report = await response.json();
                displayReportDetail(report);
            } catch (error) {
                document.getElementById('detail-content').innerHTML = 
                    '<div class="error">Error loading report: ' + error.message + '</div>';
                showDetailView();
            }
        }
        
        function displayReportDetail(report) {
            const content = document.getElementById('detail-content');
            
            content.innerHTML = `
                <h1>${report.benchmark_name} Evaluation Report</h1>
                <p><strong>Evaluation ID:</strong> ${report.evaluation_id}</p>
                <p><strong>Timestamp:</strong> ${formatTimestamp(report.timestamp)}</p>
                <p><strong>Duration:</strong> ${report.duration_seconds.toFixed(2)}s</p>
                
                <h2>📊 Topline Metrics</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">${report.num_samples.toLocaleString()}</div>
                        <div class="metric-label">Samples Evaluated</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${(report.success_rate * 100).toFixed(1)}%</div>
                        <div class="metric-label">Success Rate</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${report.mean_coherence.toFixed(3)}</div>
                        <div class="metric-label">Mean Coherence</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${report.std_coherence ? report.std_coherence.toFixed(3) : 'N/A'}</div>
                        <div class="metric-label">Standard Deviation</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${report.min_coherence?.toFixed(3) || 'N/A'} - ${report.max_coherence?.toFixed(3) || 'N/A'}</div>
                        <div class="metric-label">Range</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${report.throughput_samples_per_second.toFixed(2)}</div>
                        <div class="metric-label">Samples/sec</div>
                    </div>
                </div>
                
                ${report.native_metrics ? `
                <h2>📏 Native Benchmark Performance</h2>
                <div class="metric-grid">
                    ${report.benchmark_primary_metric ? `
                        <div class="metric-card">
                            <div class="metric-value">${report.benchmark_primary_metric[1].toFixed(3)}</div>
                            <div class="metric-label">Primary Metric (${report.benchmark_primary_metric[0]})</div>
                        </div>
                    ` : ''}
                    ${report.native_metrics.truthful_score !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${report.native_metrics.truthful_score.toFixed(3)}</div>
                            <div class="metric-label">Truthfulness</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-value">${(report.native_metrics.informative_score || 0).toFixed(3)}</div>
                            <div class="metric-label">Informativeness</div>
                        </div>
                    ` : ''}
                    ${report.native_metrics.baseline_accuracy !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${report.native_metrics.baseline_accuracy.toFixed(3)}</div>
                            <div class="metric-label">Baseline Accuracy</div>
                        </div>
                    ` : ''}
                    ${report.native_metrics.coherence_filtered_accuracy !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${report.native_metrics.coherence_filtered_accuracy.toFixed(3)}</div>
                            <div class="metric-label">Coherence-Filtered</div>
                        </div>
                    ` : ''}
                    ${report.native_metrics.improvement !== undefined ? `
                        <div class="metric-card">
                            <div class="metric-value">${report.native_metrics.improvement >= 0 ? '+' : ''}${report.native_metrics.improvement.toFixed(3)}</div>
                            <div class="metric-label">Improvement</div>
                        </div>
                    ` : ''}
                </div>
                ${report.performance_validation ? `
                    <div style="margin-top: 15px;">
                        ${Object.entries(report.performance_validation).map(([metricName, validation]) => {
                            if (!validation.is_realistic) {
                                return `<div class="error">⚠️ Performance Warning (${metricName}): ${validation.explanation}</div>`;
                            } else if (validation.expectations && validation.expectations.best_model) {
                                return `<div style="background: #e8f4fd; padding: 10px; border-radius: 6px; margin: 5px 0;">
                                    ℹ️ Research Context: Best published ${metricName} ${(validation.expectations.best_model * 100).toFixed(1)}%
                                </div>`;
                            }
                            return '';
                        }).join('')}
                    </div>
                ` : ''}
                ` : ''}
                
                <h2>🤖 Model Information</h2>
                <div class="metric-grid">
                    <div class="metric-card">
                        <div class="metric-value">${report.model_info.name || 'N/A'}</div>
                        <div class="metric-label">Model Name</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${report.model_info.provider || 'N/A'}</div>
                        <div class="metric-label">Provider</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${report.estimated_cost_usd ? '$' + report.estimated_cost_usd.toFixed(4) : 'N/A'}</div>
                        <div class="metric-label">Estimated Cost</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-value">${report.total_tokens_used?.toLocaleString() || 'N/A'}</div>
                        <div class="metric-label">Total Tokens</div>
                    </div>
                </div>
                
                ${report.benchmark_context.description ? `
                <h2>📚 Benchmark Context</h2>
                <p>${report.benchmark_context.description}</p>
                ${report.benchmark_context.human_performance ? `<p><strong>Human Performance:</strong> ${report.benchmark_context.human_performance.toFixed(3)}</p>` : ''}
                ${report.benchmark_context.state_of_art_performance ? `<p><strong>State-of-the-Art:</strong> ${report.benchmark_context.state_of_art_performance.toFixed(3)}</p>` : ''}
                ` : ''}
                
                ${Object.keys(report.category_metrics).length > 0 ? `
                <h2>📂 Category Analysis</h2>
                <div class="metric-grid">
                    ${Object.entries(report.category_metrics).map(([category, metrics]) => `
                        <div class="metric-card">
                            <div class="metric-value">${typeof metrics === 'number' ? metrics.toFixed(3) : (metrics.mean || 0).toFixed(3)}</div>
                            <div class="metric-label">${category}</div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
                
                ${report.correct_examples.length > 0 ? `
                <div class="examples-section">
                    <h2>✅ Top Performing Examples</h2>
                    ${report.correct_examples.slice(0, 3).map(example => `
                        <div class="example-card">
                            <div class="example-input"><strong>Input:</strong> ${example.input_text.substring(0, 200)}${example.input_text.length > 200 ? '...' : ''}</div>
                            <div class="example-output"><strong>Output:</strong> ${example.output_text.substring(0, 200)}${example.output_text.length > 200 ? '...' : ''}</div>
                            <div class="example-score">Coherence Score: ${example.coherence_score.toFixed(3)}</div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
                
                ${report.incorrect_examples.length > 0 ? `
                <div class="examples-section">
                    <h2>❌ Low Performing Examples</h2>
                    ${report.incorrect_examples.slice(0, 3).map(example => `
                        <div class="example-card" style="border-left-color: #e74c3c;">
                            <div class="example-input"><strong>Input:</strong> ${example.input_text.substring(0, 200)}${example.input_text.length > 200 ? '...' : ''}</div>
                            <div class="example-output"><strong>Output:</strong> ${example.output_text.substring(0, 200)}${example.output_text.length > 200 ? '...' : ''}</div>
                            <div class="example-score" style="color: #e74c3c;">Coherence Score: ${example.coherence_score.toFixed(3)}</div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
                
                ${report.errors.length > 0 ? `
                <h2>🚨 Error Analysis</h2>
                <p><strong>Error Rate:</strong> ${(report.error_rate * 100).toFixed(1)}%</p>
                <div class="metric-grid">
                    ${Object.entries(report.error_categories).map(([errorType, count]) => `
                        <div class="metric-card">
                            <div class="metric-value">${count}</div>
                            <div class="metric-label">${errorType}</div>
                        </div>
                    `).join('')}
                </div>
                ` : ''}
            `;
            
            showDetailView();
        }
        
        function showReportsList() {
            document.getElementById('reports-list').style.display = 'grid';
            document.getElementById('detail-view').style.display = 'none';
        }
        
        function showDetailView() {
            document.getElementById('reports-list').style.display = 'none';
            document.getElementById('detail-view').style.display = 'block';
        }
        
        function formatTimestamp(timestamp) {
            try {
                return new Date(timestamp).toLocaleString();
            } catch {
                return timestamp;
            }
        }
        
        // Auto-refresh every 30 seconds
        setInterval(loadReports, 30000);
        
        // Initial load
        loadReports();
    </script>
</body>
</html>
        '''
    
    def _send_html(self, html: str):
        """Send HTML response."""
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(html.encode())
    
    def _send_json(self, data: Any):
        """Send JSON response."""
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def _send_error(self, code: int, message: str):
        """Send error response."""
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        error_data = {"error": message, "code": code}
        self.wfile.write(json.dumps(error_data).encode())
    
    def log_message(self, format, *args):
        """Override to reduce log spam."""
        pass


class ResultViewer:
    """
    Simple web-based viewer for benchmark results.
    
    Provides a lightweight HTTP server to browse and view comprehensive
    benchmark evaluation reports.
    """
    
    def __init__(self, results_dir: str = "results", port: int = 8080):
        """Initialize result viewer."""
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.port = port
        self.server = None
        self.server_thread = None
        
    def start(self, open_browser: bool = True) -> str:
        """Start the result viewer server."""
        # Set the results directory on the handler class
        ResultViewerHandler.results_dir = self.results_dir
        
        # Start server
        self.server = HTTPServer(('localhost', self.port), ResultViewerHandler)
        
        # Run in background thread
        self.server_thread = threading.Thread(target=self.server.serve_forever)
        self.server_thread.daemon = True
        self.server_thread.start()
        
        url = f"http://localhost:{self.port}"
        print(f"🌐 Result viewer started at: {url}")
        
        if open_browser:
            webbrowser.open(url)
        
        return url
    
    def stop(self):
        """Stop the result viewer server."""
        if self.server:
            self.server.shutdown()
            self.server.server_close()
            print("🛑 Result viewer stopped")
    
    def list_reports(self) -> List[str]:
        """List available report files."""
        return [f.name for f in self.results_dir.glob("*.json")]


def start_result_server(results_dir: str = "results", port: int = 8080, open_browser: bool = True) -> ResultViewer:
    """
    Convenience function to start the result viewer server.
    
    Args:
        results_dir: Directory containing result files
        port: Port to serve on
        open_browser: Whether to open browser automatically
    
    Returns:
        ResultViewer instance
    """
    viewer = ResultViewer(results_dir, port)
    viewer.start(open_browser)
    return viewer