#!/bin/bash
# Health check script for Docker containers

echo "=== Docker Health Check ==="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker daemon is not running"
    exit 1
fi

echo "✅ Docker daemon is running"

# Check if containers are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Containers are running"
    docker-compose ps
else
    echo "⚠️ No containers running"
fi

# Check application health
if curl -f http://localhost:8001/ > /dev/null 2>&1; then
    echo "✅ Application is responding"
else
    echo "⚠️ Application not responding on port 8001"
fi

# Check database
if pg_isready -h localhost -p 5432 -U postgres > /dev/null 2>&1; then
    echo "✅ Database is ready"
else
    echo "⚠️ Database not ready on port 5432"
fi

# Check Ollama
if curl -f http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "✅ Ollama service is responding"
else
    echo "⚠️ Ollama not responding on port 11434"
fi

echo "=== Health Check Complete ==="
