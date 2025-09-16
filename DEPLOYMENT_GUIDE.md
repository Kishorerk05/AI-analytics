# Cloud SQL Assistant - Deployment Guide

## Prerequisites
- Docker Desktop installed and running
- Python 3.11+ (for local development)
- PostgreSQL (for local development)
- Ollama (for local development)

## Docker Deployment (Recommended)

### 1. Start Docker Services
```bash
# Ensure Docker Desktop is running, then:
docker-compose up -d
```

### 2. Verify Services
```bash
# Check all services are running
docker-compose ps

# View logs
docker-compose logs app
docker-compose logs db
docker-compose logs ollama
```

### 3. Initialize Ollama Models
```bash
# Access ollama container and pull required models
docker exec -it ollama ollama pull codellama
docker exec -it ollama ollama pull llama3.1
```

### 4. Access Application
- Frontend: http://localhost:8001
- API Documentation: http://localhost:8001/docs
- Database: localhost:5432 (postgres/123)

## Local Development Setup

### 1. Install Dependencies
```bash
pip install -r requirements_rag.txt
```

### 2. Setup PostgreSQL
```bash
# Create database and run init.sql
psql -U postgres -d postgres -f init.sql
```

### 3. Setup Ollama
```bash
# Install Ollama locally
ollama pull codellama
ollama pull llama3.1
```

### 4. Run Application
```bash
uvicorn sqlbot:app --reload --host 127.0.0.1 --port 8001
```

## Troubleshooting

### Docker Issues
- Ensure Docker Desktop is running
- Check port availability (8001, 5432, 11434)
- Clear Docker cache: `docker system prune -f`

### Application Issues
- Check logs: `docker-compose logs app`
- Verify database connection
- Ensure Ollama models are downloaded

### Performance Optimization
- Adjust worker count in Dockerfile CMD
- Increase memory limits in docker-compose.yml
- Monitor ChromaDB storage usage

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| DB_HOST | localhost | Database host |
| DB_PORT | 5432 | Database port |
| DB_NAME | postgres | Database name |
| DB_USER | postgres | Database user |
| DB_PASSWORD | 123 | Database password |
| OLLAMA_BASE_URL | http://localhost:11434 | Ollama service URL |

## Production Considerations

1. **Security**: Change default passwords
2. **SSL**: Enable HTTPS for production
3. **Monitoring**: Add logging and metrics
4. **Backup**: Regular database backups
5. **Scaling**: Consider load balancing for high traffic
