# # 



# from fastapi import FastAPI
# from fastapi.staticfiles import StaticFiles
# from fastapi.responses import FileResponse
# from pydantic import BaseModel
# import psycopg2
# import ollama
# import re
# import os

# # ---------------- PostgreSQL Config ----------------
# DB_CONFIG = {
#     "host": "localhost",
#     "port": 5432,
#     "dbname": "postgres",
#     "user": "postgres",
#     "password": "123"
# }

# SCHEMA = """
# billing(invoice_month, account_id, subscription, service, resource_group, resource_id, region, usage_qty, unit_cost, cost)
# resources(resource_id, owner, env, tags_json)
# """

# # ---------------- FastAPI Setup ----------------
# app = FastAPI(title="Cloud Cost SQL Assistant API")

# # Serve frontend folder
# app.mount("/static", StaticFiles(directory="frontend"), name="static")

# # Route to serve index.html
# @app.get("/")
# def serve_frontend():
#     index_path = os.path.join("frontend", "index.html")
#     return FileResponse(index_path)

# # ---------------- Request Model ----------------
# class QuestionRequest(BaseModel):
#     question: str

# # ---------------- NL → SQL ----------------
# def nl_to_sql(question: str) -> str:
#     prompt = f"""
#     You are a PostgreSQL SQL query generator.
#     Schema: {SCHEMA}
#     Convert the following natural language question into a valid SQL query.
#     Only output the SQL query without explanation or extra text.

#     Rules:
#     - Only use tables that exist in the database (billing, resources if exists).
#     - Do not include placeholders like <ACCOUNT_ID>, <RESOURCE_GROUP>, or [current month].
#     - If the user does not specify a month, assume the month available in the dataset.
#     - Format invoice_month exactly as it exists in the database (e.g., '2022-12').
#     - Keep SQL simple and executable.

#     Question: {question}
#     """
#     response = ollama.chat(
#         model="codellama",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response["message"]["content"].strip()

# # ---------------- Run SQL ----------------
# def run_sql(query: str):
#     try:
#         conn = psycopg2.connect(**DB_CONFIG)
#         cursor = conn.cursor()
#         cursor.execute(query)
#         rows = cursor.fetchall()
#         col_names = [desc[0] for desc in cursor.description]
#         conn.close()
#         return col_names, rows
#     except Exception as e:
#         return None, str(e)

# # ---------------- Explain SQL Results ----------------
# def explain_sql_result(question: str, sql_cols, sql_rows) -> str:
#     if not sql_rows:
#         return "There are no records matching your query."

#     prompt = f"""
#     You are an assistant that converts SQL query results into clear natural language answers.
    
#     User question: "{question}"
    
#     SQL columns: {sql_cols}
#     SQL result rows: {sql_rows}
    
#     Instructions:
#     - Provide the answer in complete sentences.
#     - Highlight the main answer.
#     - Give reasoning if applicable.
#     - Be concise and human-friendly.
#     """
    
#     response = ollama.chat(
#         model="codellama",
#         messages=[{"role": "user", "content": prompt}]
#     )
    
#     return response["message"]["content"].strip()

# # ---------------- Utility Functions ----------------
# def is_month_available(month: str) -> bool:
#     query = "SELECT DISTINCT invoice_month FROM billing;"
#     cols, rows = run_sql(query)
#     if cols is None:
#         return False
#     available_months = [r[0] for r in rows]
#     return month in available_months

# def get_single_month():
#     cols, rows = run_sql("SELECT DISTINCT invoice_month FROM billing;")
#     if cols and rows:
#         return rows[0][0]
#     return None

# def is_greeting(text: str) -> bool:
#     greetings = ["hi", "hello", "hey", "good morning", "good evening", "greetings"]
#     return text.lower() in greetings

# # ---------------- API Endpoint ----------------
# @app.post("/ask")
# def ask_question(request: QuestionRequest):
#     user_question = request.question.strip()

#     # Check for greetings
#     if is_greeting(user_question):
#         return {"bot": "Hello! I can help you with cloud cost queries. You can ask about services, usage, or costs."}

#     # Detect month in question (YYYY-MM)
#     month_match = re.search(r"\d{4}-\d{2}", user_question)
#     if month_match:
#         month_text = month_match.group(0)
#         if not is_month_available(month_text):
#             return {"bot": "Sorry, we only have data for December (2022-12)."}
#     else:
#         # If no month mentioned, append the dataset month for LLM
#         month_in_db = get_single_month()
#         if month_in_db:
#             user_question += f" for {month_in_db}"

#     # Convert NL → SQL
#     sql_query = nl_to_sql(user_question)

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg2
import os
import shutil
import ollama
import chromadb
from chromadb.config import Settings
from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from typing import List, Dict, Optional
import hashlib
import json
import logging
import traceback
import time
import re

# Import RAG system
RAG_AVAILABLE = False
get_rag_response = None
is_explanation_question = None
initialize_rag_system = None

try:
    from rag import initialize_rag_system, get_rag_response, is_explanation_question
    RAG_AVAILABLE = True
    print("RAG system imported successfully")
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"RAG system not available: {e}")
    # Define fallback functions
    def get_rag_response(question, chat_history=""):
        return "RAG system not available"
    def is_explanation_question(question):
        return False

# ---------------- PostgreSQL Config ----------------
# Get database configuration from environment variables
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", "5432")),
    "dbname": os.getenv("DB_NAME", "postgres"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123")
}

SCHEMA = """
billing(invoice_month, account_id, subscription, service, resource_group, resource_id, region, usage_qty, unit_cost, cost)
Note: Only the billing table exists. Do not use JOIN operations or reference the resources table.
"""

# ---------------- LangChain Configuration ----------------
# Initialize conversation memory
conversation_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    max_token_limit=2000
)

# Available models configuration
AVAILABLE_MODELS = {
    "llama3.1": "llama3.1:latest",
    "codellama": "codellama",
    "sql_model": "codellama",  # For SQL generation
    "conversation_model": "llama3.1:latest"  # For conversations and explanations
}

# Get Ollama base URL from environment variable
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Initialize separate LLMs for different purposes
sql_llm = Ollama(
    model=AVAILABLE_MODELS["sql_model"],  # CodeLlama for SQL generation
    base_url=OLLAMA_BASE_URL
)

conversation_llm = Ollama(
    model=AVAILABLE_MODELS["conversation_model"],  # Llama3.1 for conversations
    base_url=OLLAMA_BASE_URL
)

# Initialize RAG system
if RAG_AVAILABLE:
    try:
        # Import here to avoid circular imports
        import os
        import shutil
        
        # Clean up any corrupted ChromaDB directory
        chroma_path = "./chroma_db"
        if os.path.exists(chroma_path):
            try:
                # Try to remove if it's corrupted
                if not os.path.isdir(chroma_path):
                    os.remove(chroma_path)
                    print("🔄 Removed corrupted ChromaDB file")
            except Exception as cleanup_error:
                print(f"⚠️ ChromaDB cleanup issue: {cleanup_error}")
        
        rag_system = initialize_rag_system()
        print("🚀 RAG system initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize RAG system: {e}")
        RAG_AVAILABLE = False

# SQL Generation Prompt Template (using CodeLlama)
sql_prompt_template = PromptTemplate(
    input_variables=["schema", "question", "chat_history", "previous_results"],
    template="""
You are a PostgreSQL SQL query generator with conversation memory.

Database Schema: {schema}

Previous Conversation:
{chat_history}

Previous Query Results Context:
{previous_results}

Current Question: {question}

Instructions:
- Generate ONLY the SQL query, no explanations or markdown
- Do NOT use code blocks, backticks, or ```sql formatting
- Use conversation context to understand follow-up questions
- If user asks "what about the rest" or similar, modify the previous query appropriately
- If user refers to "previous results" or "those services", use context from chat history
- Do NOT include incomplete WHERE clauses or dangling conditions
- Do NOT add account_id filters unless specifically requested by user
- Always complete GROUP BY clauses with the appropriate columns
- Keep queries simple and executable
- For follow-up questions, build upon previous queries logically
- ONLY use the billing table - do NOT use JOIN operations
- Do NOT reference the resources table as it does not exist
- Ensure all SQL syntax is complete and valid
- Output raw SQL only, no additional text

SQL Query:
"""
)

# Result Explanation Prompt Template (using Llama3.1)
explanation_prompt_template = PromptTemplate(
    input_variables=["question", "sql_query", "results", "chat_history", "data_context"],
    template="""
You are an AI assistant that provides direct, concise answers to SQL query results.

Previous Conversation:
{chat_history}

User Question: {question}
SQL Query: {sql_query}
Query Results: {results}
Data Context: {data_context}

CRITICAL INSTRUCTIONS:
- ONLY use data that was actually returned from the SQL query
- If results are empty or insufficient, clearly state "No data available" or "Insufficient data"
- Do NOT make up numbers, dates, or comparisons that aren't in the actual results
- If asked to compare time periods and one period has no data, state this explicitly
- Never fabricate cost increases, decreases, or trends not present in the results
- Format your response as structured Markdown with this exact structure:
  1. One short **bold summary sentence**
  2. A numbered list (1..N) in sorted order: Service — $amount (2 decimals with thousands separators)
  3. A compact Markdown table with headers: service | resource_group | total_cost (2 decimals), max 10 rows
  4. If user mentioned any service that's missing, append a final note stating it's not present for the period
- No extra prose beyond these sections
- If comparison requested but data missing: "Cannot compare - only [available periods] found in dataset"

Answer:
"""
)

# Create LangChain chains with appropriate models
sql_chain = LLMChain(llm=sql_llm, prompt=sql_prompt_template, verbose=False)  # CodeLlama for SQL
explanation_chain = LLMChain(llm=conversation_llm, prompt=explanation_prompt_template, verbose=False)  # Llama3.1 for explanations

# ---------------- Logging Setup ----------------
import sys
import io

# Set stdout to handle UTF-8 encoding properly
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sqlbot.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# ---------------- FastAPI Setup ----------------
app = FastAPI(title="Cloud Cost SQL Assistant API")

# Add startup and shutdown event handlers
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI application starting up...")
    logger.info(f"RAG system available: {RAG_AVAILABLE}")
    
@app.on_event("shutdown")
async def shutdown_event():
    logger.info("FastAPI application shutting down...")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend folder
app.mount("/static", StaticFiles(directory="Frontend"), name="static")

# Route to serve index.html
@app.get("/")
def serve_frontend():
    index_path = os.path.join("Frontend", "Index.html")
    return FileResponse(index_path)

# ---------------- Request Model ----------------
class QuestionRequest(BaseModel):
    question: str

# ---------------- Enhanced NL → SQL with LangChain ----------------
def nl_to_sql_with_memory(question: str) -> str:
    """Generate SQL using LangChain with conversation memory"""
    # Get chat history
    chat_history = conversation_memory.chat_memory.messages
    chat_history_str = "\n".join([
        f"Human: {msg.content}" if isinstance(msg, HumanMessage) 
        else f"Assistant: {msg.content}" for msg in chat_history[-6:]  # Last 3 exchanges
    ])
    
    # Get previous results context
    previous_results = ""
    if len(chat_history) >= 2:
        last_ai_message = chat_history[-1] if chat_history and isinstance(chat_history[-1], AIMessage) else None
        if last_ai_message:
            previous_results = f"Last query results summary: {last_ai_message.content[:200]}..."
    
    # Generate SQL using LangChain
    try:
        sql_query = sql_chain.run(
            schema=SCHEMA,
            question=question,
            chat_history=chat_history_str,
            previous_results=previous_results
        )
        # Clean up markdown code blocks and extra text
        sql_query = clean_sql_query(sql_query)
        return sql_query.strip()
    except Exception as e:
        print(f"LangChain error, falling back to basic method: {e}")
        return nl_to_sql_fallback(question)

def clean_sql_query(sql_text: str) -> str:
    """Clean SQL query from markdown code blocks and extra text"""
    # Remove markdown code blocks
    sql_text = re.sub(r'```sql\s*', '', sql_text)
    sql_text = re.sub(r'```\s*', '', sql_text)
    
    # Remove any text after the SQL query (explanations, etc.)
    lines = sql_text.split('\n')
    sql_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Stop at explanatory text
        if any(phrase in line.lower() for phrase in ['this query', 'the query', 'explanation:', 'note:']):
            break
        sql_lines.append(line)
    
    # Join lines and clean up
    cleaned_sql = ' '.join(sql_lines)
    
    # Remove any trailing explanatory text
    cleaned_sql = re.sub(r'\s+(This|The|Note:|Explanation:).*$', '', cleaned_sql, flags=re.IGNORECASE)
    
    return cleaned_sql.strip()

def nl_to_sql_fallback(question: str) -> str:
    """Fallback method using direct ollama with CodeLlama"""
    prompt = f"""
    You are a PostgreSQL SQL query generator.
    Schema: {SCHEMA}
    Convert the following natural language question into a valid SQL query.
    Only output the SQL query without explanation or extra text.

    Rules:
    - ONLY use the billing table - do NOT use JOIN operations or reference resources table.
    - Do NOT include incomplete WHERE clauses or dangling conditions like "account_id ="
    - Do NOT add account_id filters unless specifically requested by user
    - Always complete GROUP BY clauses with the appropriate columns
    - If the user does not specify a month, assume the month available in the dataset.
    - Format invoice_month exactly as it exists in the database (e.g., '2022-12').
    - Keep SQL simple and executable.
    - Ensure all SQL syntax is complete and valid.
    - Do NOT wrap the query in markdown code blocks or backticks.

    Question: {question}
    """
    response = ollama.chat(
        model=AVAILABLE_MODELS["sql_model"],  # Use CodeLlama for SQL generation
        messages=[{"role": "user", "content": prompt}]
    )
    return clean_sql_query(response["message"]["content"])

# ---------------- Query Performance KPIs ----------------
query_performance_log = []

def analyze_join_complexity(query: str) -> Dict:
    """Analyze JOIN complexity and return KPIs"""
    query_upper = query.upper()
    
    # Count different types of JOINs
    join_types = {
        'INNER JOIN': len(re.findall(r'\bINNER\s+JOIN\b', query_upper)),
        'LEFT JOIN': len(re.findall(r'\bLEFT\s+JOIN\b', query_upper)),
        'RIGHT JOIN': len(re.findall(r'\bRIGHT\s+JOIN\b', query_upper)),
        'FULL JOIN': len(re.findall(r'\bFULL\s+JOIN\b', query_upper)),
        'CROSS JOIN': len(re.findall(r'\bCROSS\s+JOIN\b', query_upper)),
        'SIMPLE JOIN': len(re.findall(r'\bJOIN\b', query_upper)) - sum([
            len(re.findall(r'\bINNER\s+JOIN\b', query_upper)),
            len(re.findall(r'\bLEFT\s+JOIN\b', query_upper)),
            len(re.findall(r'\bRIGHT\s+JOIN\b', query_upper)),
            len(re.findall(r'\bFULL\s+JOIN\b', query_upper)),
            len(re.findall(r'\bCROSS\s+JOIN\b', query_upper))
        ])
    }
    
    total_joins = sum(join_types.values())
    
    # Count tables involved
    tables = set(re.findall(r'\bFROM\s+(\w+)|\bJOIN\s+(\w+)', query_upper))
    table_count = len([t for sublist in tables for t in sublist if t])
    
    # Complexity scoring
    complexity_score = total_joins * 2 + table_count
    if total_joins > 3:
        complexity_score += 5  # High complexity penalty
    
    complexity_level = "LOW"
    if complexity_score > 10:
        complexity_level = "HIGH"
    elif complexity_score > 5:
        complexity_level = "MEDIUM"
    
    return {
        "total_joins": total_joins,
        "join_types": join_types,
        "table_count": table_count,
        "complexity_score": complexity_score,
        "complexity_level": complexity_level,
        "has_subqueries": "SELECT" in query_upper.replace(query_upper.split("SELECT")[0] + "SELECT", "", 1)
    }

def run_sql(query: str):
    """Enhanced run_sql with performance tracking"""
    start_time = time.time()
    join_analysis = analyze_join_complexity(query)
    
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        col_names = [desc[0] for desc in cursor.description]
        conn.close()
        
        execution_time = time.time() - start_time
        
        # Log performance metrics
        performance_metrics = {
            "timestamp": time.time(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "execution_time": round(execution_time, 3),
            "row_count": len(rows),
            "success": True,
            **join_analysis
        }
        
        query_performance_log.append(performance_metrics)
        
        # Keep only last 100 queries
        if len(query_performance_log) > 100:
            query_performance_log.pop(0)
        
        return col_names, rows
        
    except Exception as e:
        execution_time = time.time() - start_time
        
        # Log failed query
        performance_metrics = {
            "timestamp": time.time(),
            "query": query[:100] + "..." if len(query) > 100 else query,
            "execution_time": round(execution_time, 3),
            "row_count": 0,
            "success": False,
            "error": str(e),
            **join_analysis
        }
        
        query_performance_log.append(performance_metrics)
        
        if len(query_performance_log) > 100:
            query_performance_log.pop(0)
        
        return None, str(e)

# ---------------- Enhanced Result Explanation with LangChain ----------------
def get_available_months() -> List[str]:
    """Get list of available months in the dataset"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT DISTINCT invoice_month FROM billing ORDER BY invoice_month")
        months = [row[0] for row in cursor.fetchall()]
        conn.close()
        return months
    except Exception as e:
        print(f"Error getting available months: {e}")
        return []

def validate_data_context(question: str, result: List) -> str:
    """Generate data context for validation"""
    available_months = get_available_months()
    
    # Check if question involves comparison
    comparison_keywords = ['vs', 'versus', 'compared to', 'compare', 'increase', 'decrease', 'difference']
    is_comparison = any(keyword in question.lower() for keyword in comparison_keywords)
    
    context = f"Available months in dataset: {available_months}. "
    
    if is_comparison:
        context += f"Comparison requested: {is_comparison}. "
        if len(available_months) < 2:
            context += "WARNING: Insufficient data for time period comparison. "
    
    context += f"Query returned {len(result)} rows. "
    
    if not result:
        context += "No data found for this query. "
    
    return context

def explain_sql_result_with_memory(question: str, sql_query: str, cols: List[str], result: List) -> str:
    """Explain SQL results using LangChain with conversation memory and data validation"""
    try:
        # Sort results deterministically before processing
        if result and len(result) > 0:
            # Determine sort columns based on available columns
            sort_result = list(result)
            if 'total_cost' in [col.lower() for col in cols] or 'cost' in [col.lower() for col in cols]:
                # Find cost column index
                cost_col_idx = None
                service_col_idx = None
                resource_group_col_idx = None
                
                for i, col in enumerate(cols):
                    if col.lower() in ['total_cost', 'cost']:
                        cost_col_idx = i
                    elif col.lower() == 'service':
                        service_col_idx = i
                    elif col.lower() == 'resource_group':
                        resource_group_col_idx = i
                
                # Sort: primary by cost DESC, tie-breakers by service ASC, resource_group ASC
                def sort_key(row):
                    cost = float(row[cost_col_idx]) if cost_col_idx is not None and row[cost_col_idx] is not None else 0
                    service = row[service_col_idx] if service_col_idx is not None and row[service_col_idx] is not None else ""
                    resource_group = row[resource_group_col_idx] if resource_group_col_idx is not None and row[resource_group_col_idx] is not None else ""
                    return (-cost, service, resource_group)
                
                sort_result.sort(key=sort_key)
            
            result = sort_result
        
        # Format results for context
        if result:
            formatted_results = f"Columns: {cols}\nRows: {result[:5]}"  # Show first 5 rows
        else:
            formatted_results = "No results returned"
        
        # Get data context for validation
        data_context = validate_data_context(question, result)
        
        # Get chat history
        chat_history = conversation_memory.chat_memory.messages
        chat_history_str = ""
        for msg in chat_history[-6:]:  # Last 3 exchanges
            if hasattr(msg, 'content'):
                role = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
                chat_history_str += f"{role}: {msg.content}\n"
        
        # Generate explanation using LangChain with data context
        explanation = explanation_chain.run(
            question=question,
            sql_query=sql_query,
            results=formatted_results,
            chat_history=chat_history_str,
            data_context=data_context
        )
        return explanation.strip()
    except Exception as e:
        print(f"LangChain explanation error, using fallback: {e}")
        return explain_sql_result_fallback(question, sql_query, cols, result)

def explain_sql_result_fallback(question: str, sql_cols, sql_rows) -> str:
    """Fallback explanation method"""
    if not sql_rows:
        return "There are no records matching your query."

    prompt = f"""
    You are an assistant that converts SQL query results into clear natural language answers.
    
    User question: "{question}"
    
    SQL columns: {sql_cols}
    SQL result rows: {sql_rows[:5]}  (showing first 5 of {len(sql_rows)} total)
    
    Instructions:
    - Provide the answer in complete sentences.
    - Highlight the main answer.
    - Give reasoning if applicable.
    - Be concise and human-friendly.
    """
    
    response = ollama.chat(
        model=AVAILABLE_MODELS["default"],
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"].strip()

# ---------------- Utility Functions ----------------
def is_month_available(month: str) -> bool:
    query = "SELECT DISTINCT invoice_month FROM billing;"
    cols, rows = run_sql(query)
    if cols is None:
        return False
    available_months = [r[0] for r in rows]
    return month in available_months

def get_single_month():
    cols, rows = run_sql("SELECT DISTINCT invoice_month FROM billing;")
    if cols and rows:
        return rows[0][0]
    return None

def is_greeting(text: str) -> bool:
    greetings = ["hi", "hello", "hey", "good morning", "good evening", "greetings"]
    return text.lower() in greetings

# ---------------- Chart Data Endpoints ----------------
@app.get("/api/charts/weekly-activity")
def get_weekly_activity_data():
    """Get cost and service data with proper representation"""
    try:
        query = """
        SELECT 
            service,
            SUM(cost) as total_cost,
            COUNT(*) as service_instances
        FROM billing 
        WHERE invoice_month = (SELECT MAX(invoice_month) FROM billing)
        GROUP BY service 
        ORDER BY total_cost DESC 
        LIMIT 8
        """
        cols, rows = run_sql(query)
        
        if cols is None:
            return {"error": "Database query failed", "details": rows}
        
        # Format data for dual-axis line chart
        services = []
        costs = []
        instances = []
        
        for row in rows:
            services.append(row[0])  # service name
            costs.append(float(row[1]) if row[1] else 0)  # total cost
            instances.append(int(row[2]) if row[2] else 0)  # service instances
        
        return {
            "labels": services,
            "datasets": [
                {
                    "label": "Total Cost ($)",
                    "data": costs,
                    "borderColor": "#00d4ff",
                    "backgroundColor": "#00d4ff33",
                    "yAxisID": "y",
                    "type": "line",
                    "tension": 0.4,
                    "fill": True
                },
                {
                    "label": "Service Instances",
                    "data": instances,
                    "borderColor": "#22c55e",
                    "backgroundColor": "#22c55e88",
                    "yAxisID": "y1",
                    "type": "bar"
                }
            ]
        }
    except Exception as e:
        return {"error": "Failed to fetch cost and service data", "details": str(e)}

@app.get("/api/charts/top-categories")
def get_top_categories_data():
    """Get service to region mapping"""
    try:
        query = """
        SELECT 
            service,
            region,
            SUM(cost) as total_cost
        FROM billing 
        WHERE invoice_month = (SELECT MAX(invoice_month) FROM billing)
        GROUP BY service, region 
        ORDER BY total_cost DESC 
        LIMIT 15
        """
        cols, rows = run_sql(query)
        
        if cols is None:
            return {"error": "Database query failed", "details": rows}
        
        # Group data by service and region
        service_region_data = {}
        regions = set()
        
        for row in rows:
            service = row[0]
            region = row[1] if row[1] else 'Unknown'
            cost = float(row[2]) if row[2] else 0
            
            if service not in service_region_data:
                service_region_data[service] = {}
            service_region_data[service][region] = cost
            regions.add(region)
        
        # Convert to chart format
        regions = sorted(list(regions))
        services = list(service_region_data.keys())[:6]  # Limit to top 6 services
        
        datasets = []
        colors = ["#00d4ff", "#22c55e", "#ef4444", "#f59e0b", "#8b5cf6", "#ec4899"]
        
        for i, region in enumerate(regions):
            data = []
            for service in services:
                cost = service_region_data.get(service, {}).get(region, 0)
                data.append(cost)
            
            datasets.append({
                "label": region,
                "data": data,
                "backgroundColor": colors[i % len(colors)] + "88",
                "borderColor": colors[i % len(colors)],
                "borderWidth": 1
            })
        
        return {
            "labels": services,
            "datasets": datasets
        }
    except Exception as e:
        return {"error": "Failed to fetch service to region data", "details": str(e)}

@app.get("/api/charts/cost-segments")
def get_cost_segments_data():
    """Get service and unit cost data"""
    try:
        query = """
        SELECT 
            service,
            AVG(unit_cost) as avg_unit_cost,
            SUM(usage_qty) as total_usage
        FROM billing 
        WHERE invoice_month = (SELECT MAX(invoice_month) FROM billing)
        AND unit_cost IS NOT NULL 
        AND usage_qty IS NOT NULL
        GROUP BY service 
        ORDER BY avg_unit_cost DESC 
        LIMIT 8
        """
        cols, rows = run_sql(query)
        
        if cols is None:
            return {"error": "Database query failed", "details": rows}
        
        services = []
        unit_costs = []
        usage_quantities = []
        
        for row in rows:
            services.append(row[0])
            unit_costs.append(float(row[1]) if row[1] else 0)
            usage_quantities.append(float(row[2]) if row[2] else 0)
        
        # Create bubble chart data or scatter plot
        bubble_data = []
        colors = ["#00d4ff", "#22c55e", "#ef4444", "#f59e0b", "#8b5cf6", "#ec4899", "#06b6d4", "#84cc16"]
        
        for i, service in enumerate(services):
            bubble_data.append({
                "x": unit_costs[i],
                "y": usage_quantities[i],
                "r": min(unit_costs[i] * usage_quantities[i] / 1000, 20),  # Bubble size based on total impact
                "label": service
            })
        
        return {
            "labels": services,
            "datasets": [{
                "label": "Service Unit Cost vs Usage",
                "data": bubble_data,
                "backgroundColor": [colors[i % len(colors)] + "66" for i in range(len(services))],
                "borderColor": [colors[i % len(colors)] for i in range(len(services))],
                "borderWidth": 2
            }]
        }
    except Exception as e:
        return {"error": "Failed to fetch service unit cost data", "details": str(e)}

# ---------------- JOIN Performance KPI Endpoints ----------------
@app.get("/api/performance/join-kpis")
def get_join_performance_kpis():
    """Get JOIN-related performance KPIs"""
    if not query_performance_log:
        return {"message": "No query performance data available yet"}
    
    # Calculate KPIs from recent queries
    recent_queries = query_performance_log[-50:]  # Last 50 queries
    
    # JOIN complexity distribution
    complexity_distribution = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    join_type_usage = {"INNER JOIN": 0, "LEFT JOIN": 0, "RIGHT JOIN": 0, "FULL JOIN": 0, "CROSS JOIN": 0, "SIMPLE JOIN": 0}
    
    total_queries = len(recent_queries)
    queries_with_joins = 0
    avg_execution_time = 0
    slow_queries = 0
    failed_queries = 0
    
    for query_log in recent_queries:
        # Complexity distribution
        complexity_distribution[query_log["complexity_level"]] += 1
        
        # JOIN type usage
        for join_type, count in query_log["join_types"].items():
            join_type_usage[join_type] += count
        
        # Performance metrics
        if query_log["total_joins"] > 0:
            queries_with_joins += 1
        
        avg_execution_time += query_log["execution_time"]
        
        if query_log["execution_time"] > 2.0:  # Queries taking more than 2 seconds
            slow_queries += 1
        
        if not query_log["success"]:
            failed_queries += 1
    
    avg_execution_time = round(avg_execution_time / total_queries, 3) if total_queries > 0 else 0
    
    # Calculate percentages
    join_usage_percentage = round((queries_with_joins / total_queries) * 100, 1) if total_queries > 0 else 0
    slow_query_percentage = round((slow_queries / total_queries) * 100, 1) if total_queries > 0 else 0
    failure_rate = round((failed_queries / total_queries) * 100, 1) if total_queries > 0 else 0
    
    return {
        "summary": {
            "total_queries_analyzed": total_queries,
            "queries_with_joins": queries_with_joins,
            "join_usage_percentage": join_usage_percentage,
            "avg_execution_time_seconds": avg_execution_time,
            "slow_queries": slow_queries,
            "slow_query_percentage": slow_query_percentage,
            "failed_queries": failed_queries,
            "failure_rate_percentage": failure_rate
        },
        "complexity_distribution": complexity_distribution,
        "join_type_usage": join_type_usage,
        "performance_alerts": {
            "high_complexity_queries": complexity_distribution["HIGH"],
            "slow_queries_count": slow_queries,
            "queries_needing_optimization": slow_queries + complexity_distribution["HIGH"]
        }
    }

@app.get("/api/performance/recent-queries")
def get_recent_query_performance():
    """Get recent query performance details"""
    if not query_performance_log:
        return {"message": "No query performance data available yet"}
    
    # Return last 20 queries with performance details
    recent_queries = query_performance_log[-20:]
    
    formatted_queries = []
    for query_log in recent_queries:
        formatted_queries.append({
            "query_preview": query_log["query"],
            "execution_time": query_log["execution_time"],
            "complexity_level": query_log["complexity_level"],
            "total_joins": query_log["total_joins"],
            "table_count": query_log["table_count"],
            "row_count": query_log["row_count"],
            "success": query_log["success"],
            "timestamp": query_log["timestamp"]
        })
    
    return {
        "recent_queries": formatted_queries,
        "total_logged": len(query_performance_log)
    }

@app.get("/api/performance/optimization-suggestions")
def get_optimization_suggestions():
    """Get query optimization suggestions based on performance data"""
    if not query_performance_log:
        return {"message": "No query performance data available yet"}
    
    suggestions = []
    recent_queries = query_performance_log[-30:]
    
    # Analyze patterns and suggest optimizations
    high_complexity_count = sum(1 for q in recent_queries if q["complexity_level"] == "HIGH")
    slow_queries = [q for q in recent_queries if q["execution_time"] > 2.0]
    failed_queries = [q for q in recent_queries if not q["success"]]
    
    if high_complexity_count > 5:
        suggestions.append({
            "type": "HIGH_COMPLEXITY",
            "message": f"Found {high_complexity_count} high-complexity queries. Consider simplifying JOINs or adding indexes.",
            "priority": "HIGH"
        })
    
    if len(slow_queries) > 3:
        suggestions.append({
            "type": "SLOW_QUERIES",
            "message": f"Found {len(slow_queries)} slow queries (>2s). Review JOIN conditions and consider query optimization.",
            "priority": "MEDIUM"
        })
    
    if len(failed_queries) > 2:
        suggestions.append({
            "type": "QUERY_FAILURES",
            "message": f"Found {len(failed_queries)} failed queries. Check JOIN syntax and table relationships.",
            "priority": "HIGH"
        })
    
    # Check for common anti-patterns
    cross_joins = sum(q["join_types"]["CROSS JOIN"] for q in recent_queries)
    if cross_joins > 0:
        suggestions.append({
            "type": "CROSS_JOIN_WARNING",
            "message": f"Found {cross_joins} CROSS JOIN operations. These can be very expensive - ensure they're intentional.",
            "priority": "HIGH"
        })
    
    return {
        "suggestions": suggestions,
        "analysis_period": f"Last {len(recent_queries)} queries"
    }

# ---------------- Model Management Endpoints ----------------
@app.get("/api/models/available")
def get_available_models():
    """Get list of available LLM models"""
    return {"models": AVAILABLE_MODELS, "current_model": AVAILABLE_MODELS["default"]}

@app.post("/api/models/switch")
def switch_model(model_name: str):
    """Switch to a different LLM model"""
    if model_name in AVAILABLE_MODELS:
        global sql_llm, conversation_llm, sql_chain, explanation_chain
        if model_name == "sql_model" or model_name == "codellama":
            sql_llm = Ollama(model=AVAILABLE_MODELS["codellama"], base_url="http://localhost:11434")
            sql_chain = LLMChain(llm=sql_llm, prompt=sql_prompt_template, verbose=False)
        elif model_name == "conversation_model" or model_name == "llama3.1":
            conversation_llm = Ollama(model=AVAILABLE_MODELS["llama3.1"], base_url="http://localhost:11434")
            explanation_chain = LLMChain(llm=conversation_llm, prompt=explanation_prompt_template, verbose=False)
        return {"message": f"Switched to model: {AVAILABLE_MODELS[model_name]}", "current_model": model_name}
    else:
        return {"error": f"Model {model_name} not available", "available_models": list(AVAILABLE_MODELS.keys())}

@app.get("/api/conversation/history")
def get_conversation_history():
    """Get current conversation history"""
    messages = conversation_memory.chat_memory.messages
    history = []
    for msg in messages:
        history.append({
            "type": "human" if isinstance(msg, HumanMessage) else "ai",
            "content": msg.content,
            "timestamp": getattr(msg, 'timestamp', None)
        })
    return {"history": history, "total_messages": len(messages)}

@app.delete("/api/conversation/clear")
def clear_conversation_history():
    """Clear conversation memory"""
    conversation_memory.clear()
    return {"message": "Conversation history cleared"}

# ---------------- Enhanced API Endpoint with LangChain ----------------
@app.post("/ask")
def ask_question(request: QuestionRequest):
    user_question = request.question.strip()
    logger.info(f"Received question: {user_question}")

    # Check for greetings
    if is_greeting(user_question):
        greeting_response = "Hello! I can help you with cloud cost queries and Azure explanations. You can ask about services, usage, costs, or technical concepts."
        logger.info("Greeting detected")
        # Add to conversation memory
        conversation_memory.chat_memory.add_user_message(user_question)
        conversation_memory.chat_memory.add_ai_message(greeting_response)
        return {"bot": greeting_response}

    # Add user question to conversation memory
    conversation_memory.chat_memory.add_user_message(user_question)

    # Check if this is an explanation/definition question - use RAG directly
    if RAG_AVAILABLE and is_explanation_question(user_question):
        logger.info(f"RAG_AVAILABLE: {RAG_AVAILABLE}, is_explanation: {is_explanation_question(user_question)}")
        logger.info("Detected explanation query, using RAG system...")
        
        # Get chat history for context
        chat_history = ""
        for msg in conversation_memory.chat_memory.messages[-6:]:  # Last 3 exchanges
            if hasattr(msg, 'content'):
                role = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
                chat_history += f"{role}: {msg.content}\n"
        
        # Get RAG response with timeout handling
        try:
            logger.info("Calling RAG system...")
            from rag import get_rag_response
            rag_response = get_rag_response(user_question, chat_history)
            logger.info(f"RAG response received: {rag_response[:100]}...")
            
            if not rag_response or rag_response.strip() == "":
                rag_response = "I don't have enough information in my knowledge base to answer that question. Please try asking about Azure services, costs, or technical concepts."
            
            conversation_memory.chat_memory.add_ai_message(rag_response)
            
            return {
                "bot": rag_response,
                "source": "RAG",
                "conversation_context": len(conversation_memory.chat_memory.messages)
            }
        except Exception as rag_error:
            logger.error(f"RAG system error: {rag_error}")
            fallback_response = "I'm having trouble accessing my knowledge base right now. Please try rephrasing your question or ask about SQL queries instead."
            conversation_memory.chat_memory.add_ai_message(fallback_response)
            return {
                "bot": fallback_response,
                "source": "RAG_ERROR",
                "conversation_context": len(conversation_memory.chat_memory.messages)
            }

    else:
        # Detect month in question (YYYY-MM)
        month_match = re.search(r"\d{4}-\d{2}", user_question)
        if month_match:
            month_text = month_match.group(0)
            if not is_month_available(month_text):
                error_response = "Sorry, we only have data for December (2022-12)."
                conversation_memory.chat_memory.add_ai_message(error_response)
                return {"bot": error_response}
        else:
            # Check if user is asking for a specific month but didn't provide YYYY-MM format
            # For now, default to December 2022 if no specific month format is provided
            if "december" in user_question.lower() or "dec" in user_question.lower():
                user_question += " for 2022-12"

    # Convert NL → SQL using LangChain with memory
    sql_query = nl_to_sql_with_memory(user_question)

    # Run SQL
    cols, result = run_sql(sql_query)
    if cols is None:
        # SQL execution failed - try RAG fallback
        logger.error(f"SQL failed with error: {result}")
        logger.info(f"RAG_AVAILABLE status: {RAG_AVAILABLE}")
        if RAG_AVAILABLE:
            logger.info("Attempting RAG fallback...")
            
            # Get chat history for context
            chat_history = ""
            for msg in conversation_memory.chat_memory.messages[-6:]:  # Last 3 exchanges
                if hasattr(msg, 'content'):
                    role = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
                    chat_history += f"{role}: {msg.content}\n"
            
            # Try RAG response as fallback
            rag_response = get_rag_response(request.question.strip(), chat_history)
            
            # If RAG provides a meaningful response, use it
            if rag_response and not rag_response.startswith("I don't have enough information"):
                conversation_memory.chat_memory.add_ai_message(rag_response)
                return {
                    "bot": rag_response,
                    "source": "RAG_FALLBACK",
                    "sql_error": result,
                    "conversation_context": len(conversation_memory.chat_memory.messages)
                }
        else:
            logger.warning("RAG system not available for fallback")
            # Try to reinitialize RAG system as a last resort
            try:
                logger.info("Attempting to reinitialize RAG system...")
                from rag import initialize_rag_system, get_rag_response
                rag_system = initialize_rag_system(force_reload=True)
                
                # Get chat history for context
                chat_history = ""
                for msg in conversation_memory.chat_memory.messages[-6:]:
                    if hasattr(msg, 'content'):
                        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "AI"
                        chat_history += f"{role}: {msg.content}\n"
                
                rag_response = get_rag_response(request.question.strip(), chat_history)
                if rag_response and not rag_response.startswith("I don't have enough information"):
                    conversation_memory.chat_memory.add_ai_message(rag_response)
                    return {
                        "bot": rag_response,
                        "source": "RAG_EMERGENCY_FALLBACK",
                        "sql_error": result,
                        "conversation_context": len(conversation_memory.chat_memory.messages)
                    }
            except Exception as emergency_error:
                logger.error(f"Emergency RAG initialization failed: {emergency_error}")
                logger.error(f"Traceback: {traceback.format_exc()}")
        
        # If RAG fallback also fails or not available, return SQL error
        error_response = f"SQL Execution Error: {result}"
        conversation_memory.chat_memory.add_ai_message(error_response)
        return {"bot": error_response, "sql": sql_query}
    else:
        # Explain results using LangChain with memory
        human_answer = explain_sql_result_with_memory(user_question, sql_query, cols, result)
        
        # Add AI response to conversation memory
        conversation_memory.chat_memory.add_ai_message(human_answer)
        
        return {
            "bot": human_answer, 
            "sql": sql_query, 
            "rows": [dict(zip(cols, r)) for r in result],
            "conversation_context": len(conversation_memory.chat_memory.messages)
        }

# ---------------- Server Startup ----------------
if __name__ == "__main__":
    import uvicorn
    try:
        logger.info("Starting FastAPI server...")
        logger.info("System Status:")
        logger.info(f"  - RAG Available: {RAG_AVAILABLE}")
        logger.info(f"  - Server Host: 0.0.0.0")
        logger.info(f"  - Server Port: 8001")
        uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise
