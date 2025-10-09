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
    groq_api_key: Optional[str] = Field(default=None, alias="GROQ_API_KEY")
    pinecone_api_key: Optional[str] = Field(default=None, alias="PINECONE_API_KEY")
    tavily_api_key: Optional[str] = Field(default=None, alias="TAVILY_API_KEY")
    
    # Model Configuration
    groq_model: str = Field(default="llama-3.1-8b-instant", alias="GROQ_MODEL")
    embedding_model: str = Field(default="all-MiniLM-L6-v2", alias="EMBEDDING_MODEL")
    
    # LLM Provider Selection
    llm_provider: str = Field(default="groq", alias="LLM_PROVIDER")  # options: "groq", "hf"
    # Hugging Face local model settings (used when llm_provider == "hf")
    hf_model_name: str = Field(default="mistralai/Mistral-7B-Instruct-v0.2", alias="HF_MODEL_NAME")
    hf_device: str = Field(default="auto", alias="HF_DEVICE")  # "auto", "cpu", "cuda", "mps"
    hf_dtype: str = Field(default="auto", alias="HF_DTYPE")  # "auto", "float16", "bfloat16", "float32"
    
    # Chunking Configuration
    chunk_size: int = Field(default=1000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=200, alias="CHUNK_OVERLAP")
    
    # Vector Database Configuration
    chroma_persist_directory: str = Field(default="./chroma_db", alias="CHROMA_PERSIST_DIRECTORY")
    collection_name: str = Field(default="pdf_documents", alias="COLLECTION_NAME")
    
    # Vector Store Selection
    vector_store_type: str = Field(default="pinecone", alias="VECTOR_STORE_TYPE")  # options: "chroma", "pinecone"
    
    # Pinecone Configuration
    pinecone_index_name: str = Field(default="pdfqa-agent", alias="PINECONE_INDEX_NAME")
    pinecone_environment: str = Field(default="gcp-starter", alias="PINECONE_ENVIRONMENT")  # Deprecated but kept for backward compatibility
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")  # aws, gcp, azure
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")  # us-east-1 works with free tier
    
    # Web Search Configuration
    enable_web_search: bool = Field(default=True, alias="ENABLE_WEB_SEARCH")  # Enable web search by default
    web_search_provider: str = Field(default="tavily", alias="WEB_SEARCH_PROVIDER")  # options: "tavily", "duckduckgo"
    max_web_results: int = Field(default=3, alias="MAX_WEB_RESULTS")
    
    # Retrieval Configuration
    retrieval_k: int = Field(default=5, alias="RETRIEVAL_K")
    similarity_threshold: float = Field(default=0.7, alias="SIMILARITY_THRESHOLD")
    
    # Logging
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    
    # UI Configuration
    app_title: str = Field(default="PDF Q/A Agent", alias="APP_TITLE")
    max_file_size_mb: int = Field(default=50, alias="MAX_FILE_SIZE_MB")
    
    # LangGraph Configuration
    execution_mode: str = Field(default="chain", alias="EXECUTION_MODE")  # options: "chain", "graph"
    checkpoint_backend: str = Field(default="memory", alias="CHECKPOINT_BACKEND")  # options: "memory", "sqlite"
    enable_human_in_loop: bool = Field(default=False, alias="ENABLE_HUMAN_IN_LOOP")
    enable_summarization: bool = Field(default=True, alias="ENABLE_SUMMARIZATION")
    max_graph_iterations: int = Field(default=3, alias="MAX_GRAPH_ITERATIONS")
    enable_tracing: bool = Field(default=False, alias="ENABLE_TRACING")
    
    model_config = {
        "env_file": ".env",
        "case_sensitive": False,
        "env_file_encoding": "utf-8",
        # Ignore extra env vars to prevent ValidationError on unknown keys
        "extra": "ignore",
    }


def get_settings() -> Settings:
    """Get application settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
