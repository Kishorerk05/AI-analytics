@echo off
echo Starting Cloud SQL Assistant Services...

REM Check if Docker Desktop is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Desktop is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo Docker is running. Starting services...

REM Start all services
docker-compose up -d

REM Wait for services to start
echo Waiting for services to initialize...
timeout /t 30 /nobreak >nul

REM Check service status
echo Checking service status...
docker-compose ps

REM Initialize Ollama models if needed
echo Initializing Ollama models...
docker exec ollama ollama pull codellama 2>nul
docker exec ollama ollama pull llama3.1 2>nul

echo.
echo Services started successfully!
echo.
echo Access points:
echo - Frontend: http://localhost:8001
echo - API Docs: http://localhost:8001/docs
echo - Database: localhost:5432 (postgres/123)
echo.
echo Press any key to open the application...
pause >nul
start http://localhost:8001
