@echo off
echo Starting Cloud SQL Assistant in Local Development Mode...
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Install dependencies if needed
echo Installing/updating dependencies...
pip install -r requirements_rag.txt

REM Start the FastAPI application
echo.
echo Starting FastAPI server...
echo Access the application at: http://127.0.0.1:8001
echo Press Ctrl+C to stop the server
echo.

uvicorn sqlbot:app --reload --host 127.0.0.1 --port 8001
