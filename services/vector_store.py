"""
Vector database service using ChromaDB and Pinecone.
Following Single Responsibility Principle: Handle only vector storage operations.
"""
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
# from langchain_huggingface import HuggingFaceEmbeddings  # commented per request to switch to OpenAI embeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
# from langchain_openai import OpenAIEmbeddings
from core.interfaces import VectorStore
from config.settings import settings
from utils.logger import get_logger

# Pinecone imports
try:
    from pinecone import Pinecone, ServerlessSpec
    from langchain_pinecone import PineconeVectorStore as LangChainPineconeVectorStore
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False

logger = get_logger(__name__)

# Helper: derive embedding dimension from OpenAI embedding model name
# _OPENAI_EMBED_DIMS = {
#     "text-embedding-3-small": 1536,
#     "text-embedding-3-large": 3072,
# }

# def _openai_embed_dim(model: str) -> int:
#     return _OPENAI_EMBED_DIMS.get(model, 1536)


class ChromaVectorStore(VectorStore):
    """ChromaDB implementation of vector store."""
    
    def __init__(self):
        # Hugging Face embeddings were previously used; switched to OpenAI per request
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=settings.embedding_model,
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        
        # if not settings.openai_api_key:
        #     raise ValueError("OPENAI_API_KEY is required for OpenAI embeddings")
        
        # self.embeddings = OpenAIEmbeddings(
        #     api_key=settings.openai_api_key,
        #     model=settings.openai_embedding_model,
        # )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.google_api_key,
            model="models/gemini-embedding-001"  # or "models/text-embedding-004"
        )
            
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_directory,
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        logger.info(f"Initialized ChromaDB collection: {settings.collection_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the vector store."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return
            
            # Prepare data for ChromaDB
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            ids = [f"{doc['metadata']['filename']}_{doc['metadata']['chunk_id']}" 
                   for doc in documents]
            
            # Generate embeddings
            embeddings = self.embeddings.embed_documents(texts)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search with query."""
        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_query(query)
            
            # Search in collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise e
    
    def delete_collection(self) -> None:
        """Delete the entire collection."""
        try:
            self.client.delete_collection(settings.collection_name)
            logger.info(f"Deleted collection: {settings.collection_name}")
        except Exception as e:
            logger.error(f"Error deleting collection: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": settings.collection_name
            }
        except Exception as e:
            logger.error(f"Error getting collection stats: {e}")
            return {"document_count": 0, "collection_name": settings.collection_name}
    
    def search_with_filter(self, query: str, metadata_filter: Dict[str, Any], 
                          k: int = 5) -> List[Dict[str, Any]]:
        """Search with metadata filtering."""
        try:
            query_embedding = self.embeddings.embed_query(query)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=metadata_filter,
                include=["documents", "metadatas", "distances"]
            )
            
            formatted_results = []
            for i in range(len(results["documents"][0])):
                formatted_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity_score": 1 - results["distances"][0][i]
                })
            
            logger.info(f"Found {len(formatted_results)} filtered results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in filtered search: {e}")
            raise


class PineconeVectorStore(VectorStore):
    """Pinecone implementation of vector store."""
    
    def __init__(self):
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone dependencies not installed. Run: pip install pinecone-client langchain-pinecone")
        
        if not settings.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY is required for Pinecone vector store")
        
        # Hugging Face embeddings were previously used; switched to OpenAI per request
        # self.embeddings = HuggingFaceEmbeddings(
        #     model_name=settings.embedding_model,
        #     model_kwargs={'device': 'cpu'},
        #     encode_kwargs={'normalize_embeddings': True}
        # )
        
        if not settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required for Google Gemini embeddings")
        
        self.embeddings = GoogleGenerativeAIEmbeddings(
            google_api_key=settings.google_api_key,
            model="models/gemini-embedding-001"  # or "models/text-embedding-004"
        )
        # Initialize Pinecone client with new API
        self.pc = Pinecone(api_key=settings.pinecone_api_key)
        
        self.index_name = settings.pinecone_index_name
        
        # Create index if it doesn't exist
        existing_indexes = [index.name for index in self.pc.list_indexes()]
        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=settings.pinecone_cloud,
                    region=settings.pinecone_region
                )
            )
            logger.info(f"Created Pinecone index: {self.index_name}")
        
        # Get the index instance
        self.index = self.pc.Index(self.index_name)
        
        # Initialize LangChain Pinecone wrapper
        self.vectorstore = LangChainPineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        logger.info(f"Initialized Pinecone vector store with index: {self.index_name}")
    
    def add_documents(self, documents: List[Dict[str, Any]]) -> None:
        """Add documents to the Pinecone vector store."""
        try:
            if not documents:
                logger.warning("No documents to add")
                return
            
            # Prepare documents for LangChain format
            texts = [doc["content"] for doc in documents]
            metadatas = [doc["metadata"] for doc in documents]
            ids = [f"{doc['metadata']['filename']}_{doc['metadata']['chunk_id']}" 
                   for doc in documents]
            
            # Add to Pinecone via LangChain wrapper
            self.vectorstore.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(documents)} documents to Pinecone vector store")
            
        except Exception as e:
            logger.error(f"Error adding documents to Pinecone vector store: {e}")
            raise
    
    def similarity_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform similarity search with query."""
        try:
            # Search using LangChain wrapper
            results = self.vectorstore.similarity_search_with_score(query, k=k)
            
            # Format results
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": 1 - score  # Convert distance to similarity
                })
            
            logger.info(f"Found {len(formatted_results)} similar documents for query")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in Pinecone similarity search: {e}")
            raise
    
    def delete_collection(self) -> None:
        """Delete the entire index."""
        try:
            self.pc.delete_index(self.index_name)
            logger.info(f"Deleted Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Error deleting Pinecone index: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the index."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "document_count": stats.get("total_vector_count", 0),
                "collection_name": self.index_name
            }
        except Exception as e:
            logger.error(f"Error getting Pinecone stats: {e}")
            return {"document_count": 0, "collection_name": self.index_name}
    
    def search_with_filter(self, query: str, metadata_filter: Dict[str, Any], 
                          k: int = 5) -> List[Dict[str, Any]]:
        """Search with metadata filtering."""
        try:
            # Convert filter format for Pinecone
            pinecone_filter = self._convert_filter(metadata_filter)
            
            results = self.vectorstore.similarity_search_with_score(
                query, k=k, filter=pinecone_filter
            )
            
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "similarity_score": 1 - score
                })
            
            logger.info(f"Found {len(formatted_results)} filtered results")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in Pinecone filtered search: {e}")
            raise
    
    def _convert_filter(self, metadata_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert generic filter to Pinecone format."""
        # Simple conversion - can be enhanced based on needs
        return metadata_filter


class VectorStoreFactory:
    """Factory for creating vector store instances following Factory Pattern."""
    
    @staticmethod
    def create_vector_store(store_type: Optional[str] = None) -> VectorStore:
        """Create vector store based on configuration."""
        store_type = store_type or settings.vector_store_type.lower()
        
        if store_type == "pinecone":
            return PineconeVectorStore()
        elif store_type == "chroma":
            return ChromaVectorStore()
        else:
            logger.warning(f"Unknown vector store type {store_type}, defaulting to ChromaDB")
            return ChromaVectorStore()
