# Development Dockerfile for coherify
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY pyproject.toml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev,benchmarks,viz]"

# Copy source code
COPY . .

# Install in development mode
RUN pip install -e .

# Set Python path
ENV PYTHONPATH=/app

# Default command
CMD ["python", "-c", "import coherify; print('Coherify loaded successfully!')"]