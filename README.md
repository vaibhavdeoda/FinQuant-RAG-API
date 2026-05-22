# FinQuant-RAG-API 🚀

A lightweight, agentic Retrieval-Augmented Generation (RAG) API designed to securely query financial reports and documents using natural language. It implements an intelligent agent workflow that forces reliance on local database contexts to eliminate model hallucinations.

---

## 📺 Application Demo

See the system ingest financial reports and answer questions in real-time.

### Watch the Demo
* **Video Walkthrough:** 📥 **[Click here to play `rag-api-demo.mov`](./rag-api-demo.mov)**
* Please download the video file to play if it doesn't play in the browser.

## 🛠️ Frameworks Used

* **FastAPI:** Core web framework providing high-performance, asynchronous endpoints and automatic Swagger documentation.
* **Microsoft Semantic Kernel:** Orchestration engine used to build the agentic workflow and manage tool/plugin callbacks seamlessly.
* **Google Gemini AI (`gemini-1.5-pro`):** The foundational Large Language Model driving reasoning and conversational context generation.
* **ChromaDB:** An ephemeral (in-memory) vector database utilizing Cosine Similarity to quickly store and retrieve relevant document chunks.
* **PyPDF:** Lightweight PDF parsing utility to seamlessly ingest financial filings and reports.

---

## ⚙️ Setup and Installation

### 1. Install Dependencies
Make sure you have your virtual environment active, then install the required framework libraries:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables
The application relies on a configuration file to load your AI foundational models securely. 

Create a file named `.env` in your project root folder and paste the following parameters into it:
```env
# Define the Google Gemini model configuration variant
GEMINI_MODEL=gemini-1.5-pro

# Your secret Google Gemini developer credential key
GEMINI_API_KEY=your_google_gemini_api_key_here
```

### 3. Add Context Documents
Place any `.txt` or `.pdf` stock reports, financial statements, or guidelines inside a directory named `data/` in the project root. The API automatically reads, chunks, and vectorizes these files on startup.

### 4. Run the API Server
Start the development server using Uvicorn:
```bash
uvicorn app.main:app --reload
```

---

## 📡 API Endpoints

Once running, you can open `http://localhost:8000/docs` to see the interactive Swagger UI, or query it using standard curl commands:

### **Ask the AI Agent**
* **Endpoint:** `POST /ask`
* **Payload:** `{"message": "What is the revenue target mentioned in the nvda reports?"}`

### **Query the Vector DB Directly**
* **Endpoint:** `POST /search`
* **Payload:** `{"query": "financial sentiment", "top_k": 3}`

### **Upload Dynamic Files**
* **Endpoint:** `POST /upload`
* Accepts multi-part form data containing `.pdf` or `.txt` files to immediately expand the agent's knowledge base.

