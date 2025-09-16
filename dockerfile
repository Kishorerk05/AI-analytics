# Use Python base image compatible with your dependencies
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for psycopg2, docx parsing, and ChromaDB
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    python3-dev \
    poppler-utils \
    curl \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements_rag.txt .
RUN pip install --no-cache-dir -r requirements_rag.txt

# Copy the application code
COPY . .

# Create necessary directories for the application
RUN mkdir -p /app/data /app/chroma_db /app/logs /app/Frontend

# Set proper permissions
RUN chmod -R 755 /app

# Expose FastAPI port
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/ || exit 1

# Run FastAPI server
CMD ["uvicorn", "sqlbot:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]
