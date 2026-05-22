from semantic_kernel.functions import kernel_function
from app.vector_db import vector_db

class FinancialPlugin:
    @kernel_function(
        description="Searches vectorized financial reports for relevant information.",
        name="read_reports"
    )
    def read_reports(self, query: str = "") -> str:
        """Search vector database for relevant financial report chunks"""
        print(f"Searching vector database for: {query}")
        return vector_db.search(query, top_k=5)