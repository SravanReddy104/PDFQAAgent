"""
Configuration settings for the PDF Q/A Agent.
Following SOLID principles - Single Responsibility for configuration management.
"""
import os
from pathlib import Path
from typing import Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Keys
    groq_api_key: str = Field(..., alias="GROQ_API_KEY")
    
    # Model Configuration
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    # Vector Database Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    collection_name: str = Field(default="pdf_documents", alias="COLLECTION_NAME")
    
    # Retrieval Configuration
    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K")
    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # UI Configuration
    app_title: str = Field(default="PDF Q/A Agent", alias="APP_TITLE")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_file_encoding": "utf-8"
    }


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
