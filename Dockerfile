# AI Project Copilot Pro - Production Dockerfile
# Cloud Run & App Engine Compatible

FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies (caching layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY streamlit_app_ENHANCED_COMPLETE_FIXED.py .
COPY agent_system_production.py .
COPY config.yaml .

# Create necessary directories
RUN mkdir -p logs data traces outputs

# Environment variables
ENV PYTHONUNBUFFERED=1

# Cloud Run port (auto-injected)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl --fail http://localhost:${PORT:-8080} || exit 1

# Start Streamlit
CMD streamlit run streamlit_app_ENHANCED_COMPLETE_FIXED.py \
    --server.port=${PORT:-8080} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --server.enableCORS=false \
    --server.enableXsrfProtection=false
