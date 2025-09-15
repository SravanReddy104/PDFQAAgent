"""
Interface definitions following SOLID principles.
Interface Segregation Principle: Define specific interfaces for different responsibilities.
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncGenerator
from pathlib import Path


class DocumentProcessor(ABC):
    """Interface for document processing operations."""
    
    @abstractmethod
    def extract_text(self, file_path: Path) -> str:
        """Extract text from a document."""
        pass
    
    @abstractmethod
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document text into smaller segments."""
        pass


class VectorStore(ABC):
    """Interface for vector database operations."""
    
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        pass
    
    @abstractmethod
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search."""
        pass
    
    @abstractmethod
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        pass


class LLMProvider(ABC):
    """Interface for LLM operations."""
    
    @abstractmethod
    async def generate_response(self, prompt: str, context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM."""
        pass
    
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts."""
        pass


class RetrieverStrategy(ABC):
    """Interface for different retrieval strategies."""
    
    @abstractmethod
    def retrieve(self, query: str, vector_store: VectorStore) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        pass


class ChunkingStrategy(ABC):
    """Interface for different chunking strategies."""
    
    @abstractmethod
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text using specific strategy."""
        pass
