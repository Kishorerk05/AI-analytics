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
from typing import List, Dict, Optional
import hashlib
import json

class RAGSystem:
    def __init__(self, chroma_db_path: str = "./chroma_db", collection_name: str = "azure_docs"):
        """Initialize RAG system with ChromaDB and Ollama models"""
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        
        # Initialize ChromaDB client
        try:
            # Ensure directory exists
            os.makedirs(chroma_db_path, exist_ok=True)
            self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
        except Exception as e:
            print(f"Collection creation failed: {e}")
            # Try alternative initialization
            try:
                import shutil
                if os.path.exists(chroma_db_path):
                    print(f"Cleaning existing ChromaDB at {chroma_db_path}")
                    shutil.rmtree(chroma_db_path)
                os.makedirs(chroma_db_path, exist_ok=True)
                self.chroma_client = chromadb.PersistentClient(path=chroma_db_path)
            except Exception as e2:
                print(f"RAG system initialization failed: {e}")
                raise e2
        
        # Initialize embedding model (nomic-embed-text)
        self.embedding_model = "nomic-embed-text"
        
        # Get Ollama base URL from environment variable
        ollama_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        # Initialize LLM for RAG responses (llama3.1)
        try:
            self.rag_llm = Ollama(
                model="llama3.1:latest",
                base_url=ollama_base_url
            )
            # Test the connection
            test_response = self.rag_llm.invoke("test")
            print("LLM connection successful")
        except Exception as e:
            print(f"LLM connection failed: {e}")
            # Fallback to a simpler model if available
            try:
                self.rag_llm = Ollama(
                    model="llama3.1",
                    base_url=ollama_base_url
                )
                print("Using fallback model: llama3.1")
            except Exception as e2:
                print(f"Fallback model also failed: {e2}")
                raise e2
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
            print(f"Connected to existing collection: {collection_name}")
        except:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                metadata={"description": "Azure cloud documentation for RAG"}
            )
            print(f"ChromaDB collection '{self.collection_name}' ready with {self.collection.count()} documents")
        
        # RAG prompt template
        self.rag_prompt = PromptTemplate(
            input_variables=["question", "context", "chat_history"],
            template="""
You are an expert Azure cloud consultant and technical documentation specialist. Use the provided context to answer the user's question accurately and comprehensively.

Previous Conversation:
{chat_history}

Context from Documentation:
{context}

Question: {question}

Instructions:
- Provide detailed explanations about Azure services, concepts, and technical details
- Use the context from the documentation to give accurate information
- If explaining Azure services, include their purpose, key features, and use cases
- Structure your response clearly with bullet points or sections when appropriate
- If the context doesn't contain enough information, clearly state what information is missing

Answer:
"""
        )
        
        # Create RAG chain using new syntax
        try:
            from langchain.chains import LLMChain
            self.rag_chain = LLMChain(llm=self.rag_llm, prompt=self.rag_prompt, verbose=False)
            print("Using LLMChain (legacy)")
        except Exception as chain_error:
            print(f"LLMChain creation failed: {chain_error}")
            # Create a simple chain alternative
            self.rag_chain = self.rag_llm
            print("Using direct LLM invocation")
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings using nomic-embed-text model"""
        embeddings = []
        for text in texts:
            try:
                response = ollama.embeddings(
                    model=self.embedding_model,
                    prompt=text
                )
                embeddings.append(response["embedding"])
            except Exception as e:
                print(f"Error generating embedding: {e}")
                # Return zero vector as fallback
                embeddings.append([0.0] * 768)  # nomic-embed-text dimension
        return embeddings
    
    def load_and_process_document(self, doc_path: str, force_reload: bool = False) -> bool:
        """Load and process document into ChromaDB"""
        try:
            # Check if document already processed
            doc_hash = self._get_file_hash(doc_path)
            metadata_key = f"doc_hash_{os.path.basename(doc_path)}"
            
            if not force_reload:
                # Check if document already exists in collection
                existing_docs = self.collection.get(
                    where={"source": os.path.basename(doc_path)}
                )
                if existing_docs['ids']:
                    print(f"Document {os.path.basename(doc_path)} already processed. Skipping...")
                    return True
            
            print(f"Processing documents in {os.path.dirname(doc_path)}...")
            
            # Load document
            if doc_path.endswith('.docx'):
                loader = Docx2txtLoader(doc_path)
                documents = loader.load()
            else:
                raise ValueError(f"Unsupported file format: {doc_path}")
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunks = []
            metadatas = []
            ids = []
            
            for i, doc in enumerate(documents):
                doc_chunks = text_splitter.split_text(doc.page_content)
                for j, chunk in enumerate(doc_chunks):
                    if len(chunk.strip()) > 50:  # Only keep substantial chunks
                        chunks.append(chunk.strip())
                        metadatas.append({
                            "source": os.path.basename(doc_path),
                            "chunk_id": j,
                            "doc_hash": doc_hash
                        })
                        ids.append(f"{os.path.basename(doc_path)}_chunk_{i}_{j}")
            
            if not chunks:
                print(f"No valid chunks found in {doc_path}")
                return False
            
            # Generate embeddings
            print(f"Generating embeddings for {len(chunks)} chunks...")
            embeddings = self.get_embeddings(chunks)
            
            # Store in ChromaDB
            self.collection.add(
                documents=chunks,
                metadatas=metadatas,
                ids=ids,
                embeddings=embeddings
            )
            
            print(f"Successfully processed and embedded {len(chunks)} document chunks from {os.path.basename(doc_path)}")
            return True
            
        except Exception as e:
            print(f"Error processing document {doc_path}: {e}")
            return False
    
    def _get_file_hash(self, file_path: str) -> str:
        """Generate hash of file for change detection"""
        hash_md5 = hashlib.md5()
        try:
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except:
            return ""
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search for relevant documents using semantic similarity"""
        try:
            # Generate query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            if results['documents'] and results['documents'][0]:
                for i, doc in enumerate(results['documents'][0]):
                    formatted_results.append({
                        "content": doc,
                        "metadata": results['metadatas'][0][i] if results['metadatas'] else {},
                        "distance": results['distances'][0][i] if results['distances'] else 1.0
                    })
            
            return formatted_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def generate_rag_response(self, question: str, chat_history: str = "") -> str:
        """Generate response using RAG"""
        try:
            print(f"Starting RAG response generation for: {question[:50]}...")
            
            # Search for relevant context
            print("Searching for relevant documents...")
            relevant_docs = self.search_documents(question, n_results=3)
            print(f"Found {len(relevant_docs)} relevant documents")
            
            if not relevant_docs:
                print("No relevant documents found")
                return "I don't have enough information in my knowledge base to answer that question. Please try asking about Azure services, costs, or technical concepts that might be covered in the documentation."
            
            # Prepare context from relevant documents
            context = "\n".join([doc["content"] for doc in relevant_docs])
            print(f"Context prepared: {len(context)} characters")
            
            # Generate response using RAG chain
            print("Generating response with LLM...")
            try:
                if hasattr(self.rag_chain, 'run'):
                    # Using LLMChain
                    response = self.rag_chain.run(
                        question=question,
                        context=context,
                        chat_history=chat_history
                    )
                else:
                    # Using direct LLM invocation
                    prompt_text = self.rag_prompt.format(
                        question=question,
                        context=context,
                        chat_history=chat_history
                    )
                    response = self.rag_chain.invoke(prompt_text)
                print("Response generated successfully")
            except Exception as llm_error:
                print(f"LLM chain error: {llm_error}")
                # Fallback to simple response
                response = f"Based on the available documentation about Azure services:\n\n{context[:500]}...\n\nFor the question '{question}', this information should help provide context."
                print("Using fallback response")
            
            return response.strip()
            
        except Exception as e:
            print(f"Error generating RAG response: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return f"I encountered an error while processing your question: {str(e)}"
    
    def is_explanation_query(self, question: str) -> bool:
        """Detect if query is asking for explanations/definitions"""
        question_lower = question.lower().strip()
        
        # Enhanced RAG keywords - conceptual/definitional questions
        rag_keywords = [
            "what is", "what are", "explain", "definition", "meaning", 
            "how does", "how do", "describe", "tell me about",
            "what does", "what means", "concept of", "understand",
            "define", "overview", "introduction", "azure services",
            "azure service", "cloud services", "cloud service",
            "types of", "kinds of", "categories of"
        ]
        
        # Azure-specific explanation patterns (always RAG)
        azure_explanation_patterns = [
            "azure services", "azure service", "what are azure",
            "types of azure", "azure offerings", "microsoft azure",
            "cloud computing", "azure resources", "azure components"
        ]
        
        # SQL keywords - data/analysis questions (these override RAG)
        sql_keywords = [
            "show", "list", "display", "get", "find", "sum", 
            "count", "average", "services by", "group by", "order by", 
            "top", "highest", "lowest", "total cost", "cost analysis"
        ]
        
        # Context-sensitive SQL patterns (only SQL if combined with specific patterns)
        sql_patterns = [
            "show total", "show cost", "list services", "compare costs",
            "total cost by", "cost by service", "services by region",
            "cost of", "usage of", "billing for", "invoice for"
        ]
        
        # Check for Azure explanation patterns first (highest priority for RAG)
        if any(pattern in question_lower for pattern in azure_explanation_patterns):
            return True
        
        # Check for specific SQL patterns (highest priority for SQL)
        if any(pattern in question_lower for pattern in sql_patterns):
            return False
            
        # Check for SQL keywords (but not if it's a conceptual question)
        if any(keyword in question_lower for keyword in sql_keywords):
            # Double-check if it's still a conceptual question
            if any(rag_keyword in question_lower for rag_keyword in rag_keywords):
                return True
            return False
        
        # Then check for RAG keywords
        if any(keyword in question_lower for keyword in rag_keywords):
            return True
            
        # Default: if starts with question words, likely RAG
        question_starters = ["what", "how", "why", "when", "where", "explain"]
        first_word = question_lower.split()[0] if question_lower.split() else ""
        
        return first_word in question_starters
    
    def initialize_documents(self, docs_folder: str = "./data") -> bool:
        """Initialize RAG system with documents from folder"""
        try:
            if not os.path.exists(docs_folder):
                print(f"Document processing failed: {docs_folder}")
                return False
            
            # Find all supported documents
            supported_extensions = ['.docx', '.pdf', '.txt']
            doc_files = []
            
            for file in os.listdir(docs_folder):
                if any(file.lower().endswith(ext) for ext in supported_extensions):
                    doc_files.append(os.path.join(docs_folder, file))
            
            if not doc_files:
                print(f"No supported documents found in {docs_folder}")
                return False
            
            print(f"Found {len(doc_files)} documents to process")
            
            # Process each document
            success_count = 0
            for doc_path in doc_files:
                if self.load_and_process_document(doc_path):
                    success_count += 1
            
            print(f"Successfully initialized RAG with {success_count}/{len(doc_files)} documents")
            return success_count > 0
            
        except Exception as e:
            print(f"RAG system initialized with {len(doc_files)} documents: {e}")
            return False
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the document collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": self.collection_name,
                "status": "ready" if count > 0 else "empty"
            }
        except Exception as e:
            return {
                "total_chunks": 0,
                "collection_name": self.collection_name,
                "status": f"error: {e}"
            }

# Global RAG instance
rag_system = None

def initialize_rag_system(force_reload: bool = False) -> RAGSystem:
    """Initialize global RAG system"""
    global rag_system
    
    if rag_system is None or force_reload:
        print("Initializing RAG system...")
        rag_system = RAGSystem()
        
        # Initialize with documents
        rag_system.initialize_documents()
        
        # Print stats
        stats = rag_system.get_collection_stats()
        print(f"RAG System Stats: {stats}")
        print("RAG system initialized successfully")
    
    return rag_system

def get_rag_response(question: str, chat_history: str = "") -> str:
    """Get RAG response for a question"""
    global rag_system
    
    try:
        if rag_system is None:
            print("Initializing RAG system for first use...")
            rag_system = initialize_rag_system()
        
        print(f"Processing RAG query: {question[:50]}...")
        response = rag_system.generate_rag_response(question, chat_history)
        print(f"RAG response generated: {len(response)} characters")
        return response
    except Exception as e:
        print(f"Error in get_rag_response: {e}")
        return f"I encountered an error while processing your question: {str(e)}"

def is_explanation_question(question: str) -> bool:
    """Check if question is asking for explanations"""
    global rag_system
    
    if rag_system is None:
        rag_system = initialize_rag_system()
    
    return rag_system.is_explanation_query(question)

# Initialize RAG system on import
if __name__ == "__main__":
    # Test the RAG system
    rag = initialize_rag_system()
    
    # Test query
    test_question = "What is a virtual machine?"
    response = get_rag_response(test_question)
    print(f"\nTest Question: {test_question}")
    print(f"RAG Response: {response}")