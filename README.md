# Cloud SQL Assistant with RAG System

A powerful AI-powered cloud cost analysis assistant that combines natural language processing with SQL query generation and Retrieval-Augmented Generation (RAG) capabilities. Built with FastAPI, PostgreSQL, ChromaDB, and Ollama LLMs.

## 🚀 Features

### Core Capabilities
- **Natural Language to SQL**: Convert plain English questions into PostgreSQL queries
- **RAG System**: Document-based question answering using ChromaDB and embeddings
- **Dual AI Models**: 
  - CodeLlama for SQL generation
  - Llama 3.1 for conversations and explanations
- **Interactive Web Interface**: Modern glassmorphism UI with real-time charts
- **Cost Analysis**: Analyze cloud billing data with intelligent insights
- **Performance Monitoring**: Query performance tracking and KPIs

### Technical Features
- **Containerized Deployment**: Docker Compose orchestration
- **Health Monitoring**: Built-in health checks for all services
- **CORS Support**: Cross-origin resource sharing enabled
- **Conversation Memory**: Context-aware conversations using LangChain
- **Document Processing**: Support for DOCX files in RAG system
- **Real-time Charts**: Interactive data visualizations

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   FastAPI       │    │   PostgreSQL    │
│   (HTML/JS)     │◄──►│   Backend       │◄──►│   Database      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌─────────────────┐    ┌─────────────────┐
                       │   Ollama LLM    │    │   ChromaDB      │
                       │   (CodeLlama +  │    │   (RAG System)  │
                       │    Llama 3.1)   │    │                 │
                       └─────────────────┘    └─────────────────┘
```

## 📋 Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- **Git** for version control
- **4GB+ RAM** recommended
- **Internet connection** for pulling Docker images and Ollama models

## 🛠️ Installation & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cloud-sql-assistant.git
cd cloud-sql-assistant
```

### 2. Project Structure

```
cloud-sql-assistant/
├── Frontend/
│   └── Index.html              # Web interface
├── data/
│   └── rag_sample_qa.docx     # Sample documents for RAG
├── chroma_db/                  # ChromaDB storage (auto-created)
├── logs/                       # Application logs (auto-created)
├── sqlbot.py                   # Main FastAPI application
├── rag.py                      # RAG system implementation
├── requirements_rag.txt        # Python dependencies
├── docker-compose.yml          # Container orchestration
├── dockerfile                  # Container build instructions
├── init.sql                    # Database initialization
├── start-services.bat          # Windows startup script
├── run-local.bat              # Local development script
├── docker-health-check.sh     # Health monitoring script
├── .dockerignore              # Docker build optimization
└── DEPLOYMENT_GUIDE.md        # Detailed deployment guide
```

### 3. Environment Configuration

The application uses environment variables defined in `docker-compose.yml`:

```yaml
environment:
  - DB_HOST=db
  - DB_PORT=5432
  - DB_NAME=postgres
  - DB_USER=postgres
  - DB_PASSWORD=123
  - OLLAMA_BASE_URL=http://ollama:11434
```

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended)

1. **Start all services:**
   ```bash
   # Windows
   start-services.bat
   
   # Linux/Mac
   docker-compose up -d
   ```

2. **Wait for initialization** (first run takes 5-10 minutes):
   - PostgreSQL database setup
   - Ollama model downloads (CodeLlama + Llama 3.1)
   - ChromaDB initialization

3. **Access the application:**
   - Frontend: http://localhost:8001
   - API Documentation: http://localhost:8001/docs
   - Database: localhost:5432 (postgres/123)

### Option 2: Local Development

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements_rag.txt
   ```

2. **Start PostgreSQL and Ollama separately**

3. **Run the application:**
   ```bash
   # Windows
   run-local.bat
   
   # Linux/Mac
   uvicorn sqlbot:app --reload --host 127.0.0.1 --port 8001
   ```

## 💡 Usage Examples

### Natural Language Queries

#### SQL Data Queries
```
"Show total cost by service"
"What are the top 5 most expensive services?"
"Show services by region"
"What's the total Azure cost for December 2022?"
```

#### RAG Explanation Queries
```
"What is a virtual machine?"
"Explain Azure Storage services"
"What are the types of cloud services?"
"How does load balancing work?"
```

### API Endpoints

#### Main Chat Endpoint
```bash
POST /ask
Content-Type: application/json

{
  "question": "Show total cost by service"
}
```

#### Chart Data Endpoints
```bash
GET /api/charts/weekly-activity     # Cost and service analysis
GET /api/charts/top-categories      # Service to region mapping
GET /api/charts/cost-segments       # Service cost segments
```

## 🔧 Configuration

### Ollama Models

The system uses two specialized models:

```python
AVAILABLE_MODELS = {
    "sql_model": "codellama",           # For SQL generation
    "conversation_model": "llama3.1:latest"  # For conversations
}
```

### Database Schema

```sql
-- Billing table structure
billing(
    invoice_month,    -- Format: 'YYYY-MM'
    account_id,       -- Account identifier
    subscription,     -- Subscription name
    service,          -- Service name
    resource_group,   -- Resource group
    resource_id,      -- Resource identifier
    region,           -- Geographic region
    usage_qty,        -- Usage quantity
    unit_cost,        -- Cost per unit
    cost              -- Total cost
)
```

### RAG System Configuration

```python
# Document processing settings
chunk_size = 1000
chunk_overlap = 200
embedding_model = "nomic-embed-text"
collection_name = "azure_docs"
```

## 🐳 Docker Services

### Service Overview

| Service | Image | Port | Purpose |
|---------|-------|------|---------|
| app | Custom (Python 3.11) | 8001 | FastAPI application |
| db | postgres:15 | 5432 | PostgreSQL database |
| ollama | ollama/ollama:latest | 11434 | LLM inference server |

### Health Checks

All services include health monitoring:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8001/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

## 📊 Monitoring & Logging

### Application Logs

```bash
# View real-time logs
docker-compose logs -f app

# View specific service logs
docker-compose logs ollama
docker-compose logs db
```

### Performance Metrics

The application tracks:
- Query execution times
- JOIN complexity analysis
- Model response times
- Database connection health

### Health Check Script

```bash
# Run manual health check
./docker-health-check.sh
```

## 🔍 Troubleshooting

### Common Issues

#### 1. Ollama Models Not Loading
```bash
# Check Ollama service
docker exec ollama ollama list

# Manually pull models
docker exec ollama ollama pull codellama
docker exec ollama ollama pull llama3.1
```

#### 2. Database Connection Issues
```bash
# Check PostgreSQL status
docker-compose ps db

# Connect to database
docker exec -it postgres-db psql -U postgres -d postgres
```

#### 3. ChromaDB Issues
```bash
# Clear ChromaDB (will reinitialize)
rm -rf chroma_db/
docker-compose restart app
```

#### 4. Port Conflicts
```bash
# Check port usage
netstat -an | findstr :8001
netstat -an | findstr :5432
netstat -an | findstr :11434

# Stop conflicting services
docker-compose down
```

## 🚀 Deployment

### Production Deployment

1. **Update environment variables:**
   ```yaml
   environment:
     - DB_PASSWORD=your_secure_password
     - OLLAMA_ORIGINS=https://yourdomain.com
   ```

2. **Use production database:**
   ```yaml
   # Replace with managed database service
   - DB_HOST=your-prod-db-host
   - DB_PORT=5432
   ```

### Scaling Considerations

- **Database**: Use managed PostgreSQL (AWS RDS, Azure Database)
- **LLM Service**: Consider GPU instances for Ollama
- **Load Balancing**: Multiple FastAPI instances behind nginx
- **Storage**: Persistent volumes for ChromaDB and logs

## 🧪 Testing

### Manual Testing Scripts

```bash
# Test API endpoints (Windows PowerShell)
./test_api.ps1

# Test RAG system (Python)
python test_rag.py
```

### API Testing Examples

```bash
# Test with curl
curl -X POST "http://localhost:8001/ask" \
     -H "Content-Type: application/json" \
     -d '{"question": "Show total cost by service"}'

# Test health endpoint
curl http://localhost:8001/
```

## 📚 API Documentation

### Interactive Documentation

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc

### Response Format

```json
{
  "bot": "Based on the data analysis...",
  "sql": "SELECT service, SUM(cost) FROM billing GROUP BY service",
  "rows": [
    ["Virtual Machines", 1250.50],
    ["Storage", 890.25]
  ],
  "source": "sql"
}
```

## 🤝 Contributing

### Development Setup

1. **Fork the repository**
2. **Create feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make changes and test**
4. **Submit pull request**


## 🙏 Acknowledgments

- **Ollama** for local LLM inference
- **ChromaDB** for vector database capabilities
- **LangChain** for LLM orchestration
- **FastAPI** for high-performance API framework
- **Chart.js** for interactive visualizations

## 📞 Support

### Getting Help

1. **Check the logs:**
   ```bash
   docker-compose logs -f
   ```

2. **Review health status:**
   ```bash
   docker-compose ps
   ```

3. **Restart services:**
   ```bash
   docker-compose restart
   ```

### Common Commands Reference

```bash
# Start services
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f app

# Restart specific service
docker-compose restart ollama

# Clean rebuild
docker-compose down -v
docker-compose build --no-cache
docker-compose up -d

# Database backup
docker exec postgres-db pg_dump -U postgres postgres > backup.sql

# Database restore
docker exec -i postgres-db psql -U postgres postgres < backup.sql
```

---

**Built  for intelligent cloud cost analysis**
