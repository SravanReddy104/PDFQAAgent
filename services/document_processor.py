"""
PDF document processing service.
Following Single Responsibility Principle: Handle only PDF processing operations.
"""
from pathlib import Path
from typing import List, Dict, Any
import pypdf
from core.interfaces import DocumentProcessor, ChunkingStrategy
from services.chunking_strategies import ChunkingStrategyFactory
from utils.logger import get_logger

logger = get_logger(__name__)


class PDFProcessor(DocumentProcessor):
    """PDF document processor with advanced text extraction."""
    
    def __init__(self, chunking_strategy: str = "hybrid"):
        self.chunking_strategy: ChunkingStrategy = ChunkingStrategyFactory.create_strategy(chunking_strategy)
    
    def extract_text(self, file_path: Path) -> str:
        """Extract text from PDF file with metadata preservation."""
        try:
            text_content = []
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip():  # Only add non-empty pages
                        # Add page marker for better chunking
                        text_content.append(f"\n--- Page {page_num + 1} ---\n{page_text}")
                
            full_text = "\n".join(text_content)
            logger.info(f"Extracted text from {len(pdf_reader.pages)} pages in {file_path.name}")
            
            return full_text
            
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise
    
    def chunk_document(self, text: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk document using the configured strategy."""
        try:
            return self.chunking_strategy.chunk_text(text, metadata)
        except Exception as e:
            logger.error(f"Error chunking document: {e}")
            raise
    
    def process_pdf(self, file_path: Path) -> List[Dict[str, Any]]:
        """Complete PDF processing pipeline."""
        try:
            # Extract text
            text = self.extract_text(file_path)
            
            # Create metadata
            metadata = {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_size": file_path.stat().st_size,
                "source_type": "pdf"
            }
            
            # Chunk document
            chunks = self.chunk_document(text, metadata)
            
            logger.info(f"Successfully processed {file_path.name} into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise


class DocumentProcessorFactory:
    """Factory for creating document processors."""
    
    @staticmethod
    def create_processor(file_type: str, **kwargs) -> DocumentProcessor:
        """Create appropriate document processor based on file type."""
        processors = {
            "pdf": PDFProcessor
        }
        
        if file_type not in processors:
            raise ValueError(f"Unsupported file type: {file_type}")
        
        return processors[file_type](**kwargs)
