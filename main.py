"""
Main application orchestrator following SOLID principles.
Dependency Inversion Principle: Depends on abstractions, not concretions.
"""
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from services import (
    PDFProcessor, 
    VectorStoreFactory,
    GroqLLMService, 
    RetrieverFactory,
    WebSearchService
)
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFQAAgent:
    """Main PDF Q/A Agent orchestrator."""
    
    def __init__(self, chunking_strategy: str = "hybrid", retrieval_strategy: str = "hybrid"):
        """Initialize the PDF Q/A Agent with configurable strategies."""
        try:
            self.pdf_processor = PDFProcessor(chunking_strategy=chunking_strategy)
            # Use vector store type from environment or settings
            vector_store_type = os.getenv('VECTOR_STORE_TYPE', settings.vector_store_type)
            self.vector_store = VectorStoreFactory.create_vector_store(store_type=vector_store_type)
            self.web_search = WebSearchService(provider="tavily")
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
            
            # Retrieve relevant documents from vector store
            relevant_docs = self.retriever.retrieve(question, self.vector_store)
            
            # Optionally enhance with web search results
            web_results = []
            if self.web_search.is_available():
                logger.info(f"Web search is available, searching for: {question}")
                web_results = self.web_search.search(question, max_results=2)
                logger.info(f"Found {len(web_results)} web search results")
                if web_results:
                    logger.info(f"Sample web result: {web_results[0].get('title', 'No title')}")
                else:
                    logger.warning("Web search returned no results")
            else:
                logger.warning("Web search is not available")
            
            if not relevant_docs:
                if web_results:
                    # Use only web search results if no documents found
                    context = self._prepare_web_context(web_results)
                    response_parts = []
                    async for chunk in self.llm_service.generate_response(question, context):
                        response_parts.append(chunk)
                    return "".join(response_parts)
                else:
                    return "I couldn't find any relevant information in the knowledge base or web search to answer your question."
            
            # Prepare context
            context = self._prepare_context(relevant_docs, web_results)
            
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
            
            # Retrieve relevant documents from vector store
            relevant_docs = self.retriever.retrieve(question, self.vector_store)
            
            # Optionally enhance with web search results
            web_results = []
            if self.web_search.is_available():
                logger.info(f"Web search is available, searching for: {question}")
                web_results = self.web_search.search(question, max_results=2)
                logger.info(f"Found {len(web_results)} web search results")
                if web_results:
                    logger.info(f"Sample web result: {web_results[0].get('title', 'No title')}")
                else:
                    logger.warning("Web search returned no results")
            else:
                logger.warning("Web search is not available")
            
            if not relevant_docs:
                if web_results:
                    # Use only web search results if no documents found
                    context = self._prepare_web_context(web_results)
                    async for chunk in self.llm_service.generate_response(question, context):
                        yield chunk
                    return
                else:
                    yield "I couldn't find any relevant information in the knowledge base or web search to answer your question."
                    return
            
            # Prepare context
            context = self._prepare_context(relevant_docs, web_results)
            
            # Stream response
            async for chunk in self.llm_service.generate_response(question, context):
                yield chunk
                
        except Exception as e:
            logger.error(f"Error in streaming question: {e}")
            yield f"An error occurred while processing your question: {str(e)}"
    
    def _prepare_context(self, relevant_docs: List[Dict[str, Any]], web_results: List[Dict[str, Any]] = None) -> str:
        """Prepare context from relevant documents."""
        context_parts = []
        
        # Add PDF document context
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
        
        # Add web search context if available
        if web_results:
            context_parts.append("\n=== WEB SEARCH RESULTS ===\n")
            for i, result in enumerate(web_results, 1):
                web_context = f"""
Web Result {i}:
Title: {result.get('title', 'Unknown')}
Source: {result.get('url', 'Unknown')}
Content: {result.get('content', '')}
---
"""
                context_parts.append(web_context)
        
        return "\n".join(context_parts)
    
    def _prepare_web_context(self, web_results: List[Dict[str, Any]]) -> str:
        """Prepare context from web search results only."""
        context_parts = ["=== WEB SEARCH RESULTS ===\n"]
        
        for i, result in enumerate(web_results, 1):
            web_context = f"""
Web Result {i}:
Title: {result.get('title', 'Unknown')}
Source: {result.get('url', 'Unknown')}
Content: {result.get('content', '')}
---
"""
            context_parts.append(web_context)
        
        return "\n".join(context_parts)
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        try:
            stats = self.vector_store.get_collection_stats()
            # Add web search info
            if hasattr(self, 'web_search'):
                stats.update(self.web_search.get_provider_info())
            return stats
        except Exception as e:
            logger.error(f"Error getting knowledge base stats: {e}")
            return {"document_count": 0, "collection_name": "unknown"}
    
    def clear_knowledge_base(self) -> bool:
        """Clear the entire knowledge base."""
        try:
            self.vector_store.delete_collection()
            # Reinitialize vector store
            self.vector_store = VectorStoreFactory.create_vector_store()
            logger.info("Knowledge base cleared successfully")
            return True
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {e}")
            return False
