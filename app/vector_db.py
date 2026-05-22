import os
import chromadb
from pypdf import PdfReader
from typing import List

class VectorDB:
    def __init__(self):
        """Initialize in-memory Chroma vector database"""
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(
            name="financial_reports",
            metadata={"hnsw:space": "cosine"}
        )
        self.doc_id_counter = 0
    
    def add_document(self, filename: str, content: str) -> dict:
        """Add document to vector database by chunking and embedding"""
        try:
            # Split content into chunks (around 500 chars each)
            chunks = self._chunk_content(content, chunk_size=500)
            
            ids = []
            for i, chunk in enumerate(chunks):
                doc_id = f"{filename}_{i}"
                self.collection.add(
                    documents=[chunk],
                    metadatas=[{"source": filename, "chunk": i}],
                    ids=[doc_id]
                )
                ids.append(doc_id)
            
            return {"status": "success", "filename": filename, "chunks": len(ids)}
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    def search(self, query: str, top_k: int = 5) -> str:
        """Search vector database and return relevant documents"""
        try:
            if self.collection.count() == 0:
                return "No documents in vector database."
            
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No relevant documents found."
            
            # Combine search results with source information
            output = "Relevant Information from Documents:\n\n"
            for i, (doc, metadata) in enumerate(zip(results["documents"][0], results["metadatas"][0])):
                source = metadata.get("source", "Unknown")
                output += f"[{source}]\n{doc}\n\n"
            
            return output
        except Exception as e:
            return f"Error searching database: {str(e)}"
    
    def flexible_search(self, query: str, top_k: int = 5, where: dict = None, include: list = None) -> dict:
        """Flexible search with Chroma query features"""
        try:
            if self.collection.count() == 0:
                return {"status": "empty", "message": "No documents in vector database."}
            
            if include is None:
                include = ["documents", "metadatas", "distances"]
            
            # Build query kwargs
            query_kwargs = {
                "query_texts": [query],
                "n_results": top_k,
                "include": include
            }
            
            # Add where filter if provided
            if where:
                query_kwargs["where"] = where
            
            results = self.collection.query(**query_kwargs)
            
            if not results["documents"] or not results["documents"][0]:
                return {"status": "no_results", "message": "No relevant documents found."}
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                result_item = {}
                
                if "documents" in include:
                    result_item["document"] = results["documents"][0][i]
                
                if "metadatas" in include:
                    result_item["metadata"] = results["metadatas"][0][i]
                
                if "distances" in include:
                    result_item["distance"] = results["distances"][0][i]
                
                if "embeddings" in include and "embeddings" in results:
                    result_item["embedding"] = results["embeddings"][0][i]
                
                formatted_results.append(result_item)
            
            return {
                "status": "success",
                "query": query,
                "results_count": len(formatted_results),
                "results": formatted_results
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def _chunk_content(self, content: str, chunk_size: int = 500) -> List[str]:
        """Split content into overlapping chunks"""
        chunks = []
        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            if chunk.strip():  # Only add non-empty chunks
                chunks.append(chunk)
        return chunks
    
    def get_stats(self) -> dict:
        """Get vector database statistics"""
        return {
            "total_documents": self.collection.count(),
            "collection_name": self.collection.name
        }

# Global vector database instance
vector_db = VectorDB()
