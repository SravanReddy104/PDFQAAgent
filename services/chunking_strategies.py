"""
Advanced chunking strategies implementation.
Following Open/Closed Principle: Easy to extend with new chunking strategies.
"""
from typing import List, Dict, Any
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from core.interfaces import ChunkingStrategy
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Recursive character-level chunking strategy."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text using recursive character splitting."""
        try:
            chunks = self.text_splitter.split_text(text)
            
            chunked_docs = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "chunking_strategy": "recursive"
                }
                chunked_docs.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunked_docs)} chunks using recursive strategy")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error in recursive chunking: {e}")
            raise


class SemanticChunkingStrategy(ChunkingStrategy):
    """Semantic chunking strategy based on sentence embeddings."""
    
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model
        )
        self.semantic_chunker = SemanticChunker(
            embeddings=self.embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=95
        )
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text using semantic similarity."""
        try:
            chunks = self.semantic_chunker.split_text(text)
            
            chunked_docs = []
            for i, chunk in enumerate(chunks):
                chunk_metadata = {
                    **metadata,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "chunking_strategy": "semantic"
                }
                chunked_docs.append({
                    "content": chunk,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunked_docs)} chunks using semantic strategy")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}")
            raise


class ContextualChunkingStrategy(ChunkingStrategy):
    """Contextual chunking with document context preservation."""
    
    def __init__(self):
        self.base_chunker = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len
        )
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk text with contextual information."""
        try:
            # Create document summary for context
            doc_summary = self._create_document_summary(text, metadata)
            
            # Split into base chunks
            base_chunks = self.base_chunker.split_text(text)
            
            chunked_docs = []
            for i, chunk in enumerate(base_chunks):
                # Add contextual information to each chunk
                contextual_chunk = self._add_context_to_chunk(
                    chunk, doc_summary, i, len(base_chunks)
                )
                
                chunk_metadata = {
                    **metadata,
                    "chunk_id": i,
                    "chunk_size": len(contextual_chunk),
                    "chunking_strategy": "contextual",
                    "has_context": True
                }
                
                chunked_docs.append({
                    "content": contextual_chunk,
                    "metadata": chunk_metadata
                })
            
            logger.info(f"Created {len(chunked_docs)} chunks using contextual strategy")
            return chunked_docs
            
        except Exception as e:
            logger.error(f"Error in contextual chunking: {e}")
            raise
    
    def _create_document_summary(self, text: str, metadata: Dict[str, Any]) -> str:
        """Create a brief summary of the document for context."""
        # Simple extractive summary - first few sentences
        sentences = text.split('. ')[:3]
        summary = '. '.join(sentences)
        
        doc_info = f"Document: {metadata.get('filename', 'Unknown')}"
        if 'page_number' in metadata:
            doc_info += f", Page: {metadata['page_number']}"
        
        return f"{doc_info}\nSummary: {summary}"
    
    def _add_context_to_chunk(self, chunk: str, doc_summary: str, 
                             chunk_idx: int, total_chunks: int) -> str:
        """Add contextual information to a chunk."""
        context_header = (
            f"[Context: This is chunk {chunk_idx + 1} of {total_chunks} from the document]\n"
            f"{doc_summary}\n"
            f"[Chunk Content:]\n"
        )
        return context_header + chunk


class HybridChunkingStrategy(ChunkingStrategy):
    """Hybrid strategy combining multiple chunking approaches."""
    
    def __init__(self):
        self.recursive_strategy = RecursiveChunkingStrategy()
        self.semantic_strategy = SemanticChunkingStrategy()
    
    def chunk_text(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Use hybrid approach for optimal chunking."""
        try:
            # Use semantic chunking for longer documents, recursive for shorter ones
            text_length = len(text)
            
            if text_length > 10000:  # Use semantic for long documents
                chunks = self.semantic_strategy.chunk_text(text, metadata)
                logger.info("Used semantic chunking for long document")
            else:  # Use recursive for shorter documents
                chunks = self.recursive_strategy.chunk_text(text, metadata)
                logger.info("Used recursive chunking for shorter document")
            
            # Update metadata to reflect hybrid strategy
            for chunk in chunks:
                chunk["metadata"]["chunking_strategy"] = "hybrid"
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error in hybrid chunking: {e}")
            # Fallback to recursive chunking
            return self.recursive_strategy.chunk_text(text, metadata)


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""
    
    @staticmethod
    def create_strategy(strategy_type: str) -> ChunkingStrategy:
        """Create a chunking strategy based on type."""
        strategies = {
            "recursive": RecursiveChunkingStrategy,
            "semantic": SemanticChunkingStrategy,
            "contextual": ContextualChunkingStrategy,
            "hybrid": HybridChunkingStrategy
        }
        
        if strategy_type not in strategies:
            logger.warning(f"Unknown strategy {strategy_type}, using recursive")
            strategy_type = "recursive"
        
        return strategies[strategy_type]()
