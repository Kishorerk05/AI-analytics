# PowerShell API Test Script for RAG Integration System
# This script provides multiple ways to test the /ask endpoint without JSON escape issues

Write-Host "=== RAG Integration API Test Script ===" -ForegroundColor Green
Write-Host "Server URL: http://127.0.0.1:8001" -ForegroundColor Yellow
Write-Host ""

# Method 1: Using PowerShell hashtable with ConvertTo-Json (Recommended)
Write-Host "Method 1: PowerShell Hashtable -> JSON" -ForegroundColor Cyan
Write-Host "Command:" -ForegroundColor Gray
Write-Host '$body = @{ question = "What is a virtual machine?" } | ConvertTo-Json'
Write-Host 'Invoke-RestMethod -Uri "http://127.0.0.1:8001/ask" -Method POST -ContentType "application/json" -Body $body'
Write-Host ""

try {
    $body1 = @{ question = "What is a virtual machine?" } | ConvertTo-Json
    $response1 = Invoke-RestMethod -Uri "http://127.0.0.1:8001/ask" -Method POST -ContentType "application/json" -Body $body1
    Write-Host "✅ Response received:" -ForegroundColor Green
    Write-Host "Bot: $($response1.bot)" -ForegroundColor White
    Write-Host "Source: $($response1.source)" -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host "❌ Error with Method 1: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# Method 2: Using single-quoted JSON string (Alternative)
Write-Host "Method 2: Single-Quoted JSON String" -ForegroundColor Cyan
Write-Host "Command:" -ForegroundColor Gray
Write-Host "Invoke-RestMethod -Uri 'http://127.0.0.1:8001/ask' -Method POST -ContentType 'application/json' -Body '{\"question\":\"What is Azure Storage?\"}'"
Write-Host ""

try {
    $response2 = Invoke-RestMethod -Uri 'http://127.0.0.1:8001/ask' -Method POST -ContentType 'application/json' -Body '{"question":"What is Azure Storage?"}'
    Write-Host "✅ Response received:" -ForegroundColor Green
    Write-Host "Bot: $($response2.bot)" -ForegroundColor White
    Write-Host "Source: $($response2.source)" -ForegroundColor Gray
    Write-Host ""
} catch {
    Write-Host "❌ Error with Method 2: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

# Method 3: Using here-string for complex queries
Write-Host "Method 3: Here-String for Complex Queries" -ForegroundColor Cyan
Write-Host "Command:" -ForegroundColor Gray
Write-Host '$jsonBody = @"'
Write-Host '{"question":"Show total cost by service"}'
Write-Host '"@'
Write-Host 'Invoke-RestMethod -Uri "http://127.0.0.1:8001/ask" -Method POST -ContentType "application/json" -Body $jsonBody'
Write-Host ""

try {
    $jsonBody = @"
{"question":"Show total cost by service"}
"@
    $response3 = Invoke-RestMethod -Uri "http://127.0.0.1:8001/ask" -Method POST -ContentType "application/json" -Body $jsonBody
    Write-Host "✅ Response received:" -ForegroundColor Green
    Write-Host "Bot: $($response3.bot)" -ForegroundColor White
    Write-Host "Source: $($response3.source)" -ForegroundColor Gray
    if ($response3.rows) {
        Write-Host "Rows returned: $($response3.rows.Count)" -ForegroundColor Gray
    }
    Write-Host ""
} catch {
    Write-Host "❌ Error with Method 3: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host ""
}

Write-Host "=== Test Types ===" -ForegroundColor Green
Write-Host "RAG Queries (explanation questions):" -ForegroundColor Yellow
Write-Host "  • What is a virtual machine?"
Write-Host "  • Explain Azure Storage"
Write-Host "  • What does load balancing mean?"
Write-Host ""
Write-Host "SQL Queries (data questions):" -ForegroundColor Yellow
Write-Host "  • Show total cost by service"
Write-Host "  • Show services by region"
Write-Host "  • What are the top 5 most expensive services?"
Write-Host ""

Write-Host "=== Quick Test Commands ===" -ForegroundColor Green
Write-Host ""
Write-Host "# Test RAG System:" -ForegroundColor Cyan
Write-Host '$body = @{ question = "What is a virtual machine?" } | ConvertTo-Json; Invoke-RestMethod -Uri "http://127.0.0.1:8001/ask" -Method POST -ContentType "application/json" -Body $body'
Write-Host ""
Write-Host "# Test SQL System:" -ForegroundColor Cyan
Write-Host '$body = @{ question = "Show total cost by service" } | ConvertTo-Json; Invoke-RestMethod -Uri "http://127.0.0.1:8001/ask" -Method POST -ContentType "application/json" -Body $body'
Write-Host ""
Write-Host "# Test with single quotes (no escaping needed):" -ForegroundColor Cyan
Write-Host "Invoke-RestMethod -Uri 'http://127.0.0.1:8001/ask' -Method POST -ContentType 'application/json' -Body '{\"question\":\"What is Azure?\"}'"
Write-Host ""

Write-Host "=== Server Status Check ===" -ForegroundColor Green
try {
    $healthCheck = Invoke-RestMethod -Uri "http://127.0.0.1:8001/docs" -Method GET
    Write-Host "✅ Server is running and accessible" -ForegroundColor Green
} catch {
    Write-Host "❌ Server not accessible. Make sure it's running on port 8001" -ForegroundColor Red
    Write-Host "Start server with: uvicorn sqlbot:app --reload --host 127.0.0.1 --port 8001" -ForegroundColor Yellow
}
