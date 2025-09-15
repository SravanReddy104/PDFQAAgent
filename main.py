"""
Main application orchestrator following SOLID principles.
Dependency Inversion Principle: Depends on abstractions, not concretions.
"""
from pathlib import Path
from typing import List, Dict, Any, Optional
import asyncio
from services import (
    PDFProcessor, 
    ChromaVectorStore, 
    GroqLLMService, 
    RetrieverFactory
)
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFQAAgent:
    """Main PDF Q/A Agent orchestrator."""
    
    def __init__(self, chunking_strategy: str = "hybrid", retrieval_strategy: str = "hybrid"):
        """Initialize the PDF Q/A Agent with configurable strategies."""
        try:
            self.pdf_processor = PDFProcessor(chunking_strategy=chunking_strategy)
            self.vector_store = ChromaVectorStore()
            self.llm_service = GroqLLMService()
            self.retriever = RetrieverFactory.create_retriever(retrieval_strategy)
            
            logger.info("PDF Q/A Agent initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing PDF Q/A Agent: {e}")
            raise
    
    async def process_pdf(self, file_path: Path) -> bool:
        """Process a PDF file and add it to the knowledge base."""
        try:
            logger.info(f"Processing PDF: {file_path.name}")
            
            # Process PDF into chunks
            chunks = self.pdf_processor.process_pdf(file_path)
            
            if not chunks:
                logger.warning(f"No chunks extracted from {file_path.name}")
                return False
            
            # Add to vector store
            self.vector_store.add_documents(chunks)
            
            logger.info(f"Successfully processed and stored {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            return False
    
    async def ask_question(self, question: str) -> str:
        """Ask a question and get a streaming response."""
        try:
            logger.info(f"Processing question: {question[:100]}...")
            
            # Retrieve relevant documents
            relevant_docs = self.retriever.retrieve(question, self.vector_store)
            
            if not relevant_docs:
                return "I couldn't find any relevant information in the knowledge base to answer your question."
            
            # Prepare context
            context = self._prepare_context(relevant_docs)
            
            # Generate response
            response_parts = []
            async for chunk in self.llm_service.generate_response(question, context):
                response_parts.append(chunk)
            
            full_response = "".join(response_parts)
            logger.info("Question answered successfully")
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error answering question: {e}")
            return f"An error occurred while processing your question: {str(e)}"
    
    async def ask_question_stream(self, question: str):
        """Ask a question and get a streaming response generator."""
        try:
            logger.info(f"Processing streaming question: {question[:100]}...")
            
            # Retrieve relevant documents
            relevant_docs = self.retriever.retrieve(question, self.vector_store)
            
            if not relevant_docs:
                yield "I couldn't find any relevant information in the knowledge base to answer your question."
                return
            
            # Prepare context
            context = self._prepare_context(relevant_docs)
            
            # Stream response
            async for chunk in self.llm_service.generate_response(question, context):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming question: {e}")
            yield f"An error occurred while processing your question: {str(e)}"
    
    def _prepare_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """Prepare context from relevant documents."""
        context_parts = []
        
        for i, doc in enumerate(relevant_docs, 1):
            metadata = doc["metadata"]
            content = doc["content"]
            similarity_score = doc.get("similarity_score", 0)
            
            context_part = f"""
Document {i}:
Source: {metadata.get('filename', 'Unknown')}
Relevance Score: {similarity_score:.3f}
Content: {content}
---
"""
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            return self.vector_store.get_collection_stats()
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {"document_count": 0, "collection_name": "unknown"}
    
    def clear_knowledge_base(self) -> bool:
        """Clear the entire knowledge base."""
        try:
            self.vector_store.delete_collection()
            # Reinitialize vector store
            self.vector_store = ChromaVectorStore()
            logger.info("Knowledge base cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False
