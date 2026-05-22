import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.functions import KernelArguments
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatCompletion, GoogleAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from pypdf import PdfReader

load_dotenv()


from app.plugins import FinancialPlugin
from app.vector_db import vector_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events"""
    # Startup
    print("Initializing vector database with existing documents...")
    
    if os.path.exists("data"):
        vectorized_count = 0
        for filename in os.listdir("data"):
            filepath = f"data/{filename}"
            
            if not os.path.isfile(filepath):
                continue
            
            try:
                content = ""
                
                if filename.endswith(".txt"):
                    with open(filepath, "r") as f:
                        content = f.read()
                elif filename.endswith(".pdf"):
                    pdf_reader = PdfReader(filepath)
                    for page in pdf_reader.pages:
                        content += page.extract_text() + "\n"
                else:
                    continue
                
                if content.strip():
                    result = vector_db.add_document(filename, content)
                    if result["status"] == "success":
                        vectorized_count += 1
                        print(f"Vectorized: {filename} ({result['chunks']} chunks)")
            except Exception as e:
                print(f"Error vectorizing {filename}: {str(e)}")
        
        stats = vector_db.get_stats()
        print(f"Startup complete. Total documents in vector DB: {stats['total_documents']}")
    else:
        print("No data folder found. Skipping document vectorization.")
    
    yield
    
    # Shutdown (optional cleanup)
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)

# 1. Initialize Kernel and add Gemini
kernel = Kernel()
kernel.add_service(
    GoogleAIChatCompletion(
        service_id="gemini-service",
        gemini_model_id=os.getenv("GEMINI_MODEL"),
        api_key=os.getenv("GEMINI_API_KEY"),
    )
)

# 2. Register our Local File Plugin
kernel.add_plugin(FinancialPlugin(), plugin_name="Finance")

class UserQuery(BaseModel):
    message: str

class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    where: dict = None  # Metadata filter
    include: list = ["documents", "metadatas", "distances"]  # What to return

@app.post("/ask")
async def ask_agent(query: UserQuery):
    # 4. Define Gemini tool choices FIRST
    execution_settings = GoogleAIChatPromptExecutionSettings(service_id="gemini-service")
    execution_settings.function_choice_behavior = FunctionChoiceBehavior.Required(
        included_functions=["Finance-read_reports"],
        auto_invoke=True
    )
    
    # 3. Setup the agent with execution settings to restrict its behavior
    agent = ChatCompletionAgent(
        kernel=kernel,
        name="FinanceAgent",
        instructions="You are a financial analysis assistant. You MUST ONLY use the Finance-read_reports function to retrieve relevant information from the vectorized financial documents. Do NOT search the internet or use any other sources. Always provide answers based on the retrieved document context.",
        execution_settings=execution_settings
    )

    # 6. Initialize ChatHistory and append the user's message
    history = ChatHistory()
    history.add_user_message(query.message)

    # 7. Stream the agent invocation loop
    response_text = ""
    async for message in agent.invoke(history=history):
        response_text += str(message.content)
    
    return {"answer": response_text}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the data folder and vectorize it"""
    try:
        # Ensure data folder exists
        os.makedirs("data", exist_ok=True)
        
        # Save the file to the data folder
        file_path = f"data/{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract content based on file type
        content = ""
        if file.filename.endswith(".txt"):
            with open(file_path, "r") as f:
                content = f.read()
        elif file.filename.endswith(".pdf"):
            pdf_reader = PdfReader(file_path)
            for page in pdf_reader.pages:
                content += page.extract_text() + "\n"
        else:
            return {"error": "Only .txt and .pdf files are supported"}, 400
        
        # Vectorize and add to database
        result = vector_db.add_document(file.filename, content)
        
        if result["status"] == "success":
            return {
                "message": f"File '{file.filename}' uploaded and vectorized successfully",
                "path": file_path,
                "chunks_created": result["chunks"],
                "db_stats": vector_db.get_stats()
            }
        else:
            return {"error": result["message"]}, 400
            
    except Exception as e:
        return {"error": str(e)}, 400

@app.get("/vector-db/stats")
async def get_vector_db_stats():
    """Get vector database statistics"""
    return vector_db.get_stats()

@app.post("/search")
async def search_vector_db(request: SearchRequest):
    """
    Flexible query endpoint for Chroma vector database.
    
    Query parameters:
    - query: Search text
    - top_k: Number of results (default: 5)
    - where: Metadata filter (dict) - e.g., {"source": "aapl.txt"}
    - include: What to include in results - ["documents", "metadatas", "distances", "embeddings"] (default: ["documents", "metadatas", "distances"])
    
    Example curl:
    curl -X POST "http://localhost:8001/search" \
      -H "Content-Type: application/json" \
      -d '{"query": "revenue", "top_k": 3, "include": ["documents", "metadatas", "distances"]}'
    """
    try:
        result = vector_db.flexible_search(
            query=request.query,
            top_k=request.top_k,
            where=request.where,
            include=request.include
        )
        return result
    except Exception as e:
        return {"error": str(e)}, 400