"""
Agent Factory for creating different types of RAG agents.
Provides choice between traditional chain-based and LangGraph-based execution.
"""

import os
from typing import Optional, Dict, Any
from enum import Enum
import uuid

from config.settings import settings
from main import PDFQAAgent  # Traditional chain-based agent
from services.langgraph_rag import LangGraphRAGAgent
from utils.logger import get_logger

logger = get_logger(__name__)


class ExecutionMode(Enum):
    """Execution modes for RAG agents."""
    CHAIN = "chain"
    GRAPH = "graph"


class AgentFactory:
    """Factory for creating RAG agents with different execution modes."""
    
    @staticmethod
    def create_agent(
        execution_mode: Optional[str] = None,
        vector_store_type: Optional[str] = None,
        chunking_strategy: str = "hybrid",
        retrieval_strategy: str = "hybrid",
        # LangGraph-specific options
        checkpoint_backend: str = "memory",
        enable_human_in_loop: bool = False,
        enable_summarization: bool = True,
        max_iterations: int = 3,
        **kwargs
    ):
        """
        Create a RAG agent based on execution mode.
        
        Args:
            execution_mode: "chain" or "graph"
            vector_store_type: Vector store type ("chroma" or "pinecone")
            chunking_strategy: Document chunking strategy
            retrieval_strategy: Document retrieval strategy
            checkpoint_backend: "memory" or "sqlite" (graph mode only)
            enable_human_in_loop: Enable human-in-the-loop (graph mode only)
            enable_summarization: Enable response summarization (graph mode only)
            max_iterations: Maximum workflow iterations (graph mode only)
        
        Returns:
            RAG agent instance
        """
        # Determine execution mode
        mode = execution_mode or os.getenv('EXECUTION_MODE', 'chain')
        mode = mode.lower()
        
        # Determine vector store type
        vector_store = vector_store_type or os.getenv('VECTOR_STORE_TYPE', settings.vector_store_type)
        
        logger.info(f"Creating agent with mode: {mode}, vector_store: {vector_store}")
        
        if mode == ExecutionMode.GRAPH.value:
            return AgentFactory._create_graph_agent(
                vector_store_type=vector_store,
                chunking_strategy=chunking_strategy,
                retrieval_strategy=retrieval_strategy,
                checkpoint_backend=checkpoint_backend,
                enable_human_in_loop=enable_human_in_loop,
                enable_summarization=enable_summarization,
                max_iterations=max_iterations,
                **kwargs
            )
        else:
            return AgentFactory._create_chain_agent(
                vector_store_type=vector_store,
                chunking_strategy=chunking_strategy,
                retrieval_strategy=retrieval_strategy,
                **kwargs
            )
    
    @staticmethod
    def _create_chain_agent(
        vector_store_type: str,
        chunking_strategy: str,
        retrieval_strategy: str,
        **kwargs
    ) -> PDFQAAgent:
        """Create traditional chain-based agent."""
        try:
            # Set vector store type in environment for the agent
            os.environ['VECTOR_STORE_TYPE'] = vector_store_type
            
            agent = PDFQAAgent(
                chunking_strategy=chunking_strategy,
                retrieval_strategy=retrieval_strategy
            )
            
            logger.info("Chain-based RAG agent created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating chain agent: {e}")
            raise
    
    @staticmethod
    def _create_graph_agent(
        vector_store_type: str,
        chunking_strategy: str,
        retrieval_strategy: str,
        checkpoint_backend: str,
        enable_human_in_loop: bool,
        enable_summarization: bool,
        max_iterations: int,
        **kwargs
    ) -> LangGraphRAGAgent:
        """Create LangGraph-based agent."""
        try:
            agent = LangGraphRAGAgent(
                vector_store_type=vector_store_type,
                chunking_strategy=chunking_strategy,
                retrieval_strategy=retrieval_strategy,
                checkpoint_backend=checkpoint_backend,
                enable_human_in_loop=enable_human_in_loop,
                enable_summarization=enable_summarization,
                max_iterations=max_iterations
            )
            
            logger.info("LangGraph-based RAG agent created successfully")
            return agent
            
        except Exception as e:
            logger.error(f"Error creating graph agent: {e}")
            raise
    
    @staticmethod
    def get_available_modes() -> Dict[str, Dict[str, Any]]:
        """Get information about available execution modes."""
        return {
            "chain": {
                "name": "Chain Mode",
                "description": "Traditional sequential processing",
                "features": [
                    "Fast execution",
                    "Lightweight",
                    "Simple workflow",
                    "Direct LLM calls"
                ],
                "best_for": [
                    "Simple Q&A",
                    "Quick responses",
                    "Resource-constrained environments"
                ]
            },
            "graph": {
                "name": "Graph Mode",
                "description": "LangGraph workflow with state management",
                "features": [
                    "State persistence",
                    "Error recovery",
                    "Human-in-the-loop",
                    "Workflow observability",
                    "Conditional execution",
                    "Middleware support"
                ],
                "best_for": [
                    "Complex workflows",
                    "Multi-step reasoning",
                    "Production environments",
                    "Debugging and monitoring"
                ]
            }
        }
    
    @staticmethod
    def get_mode_recommendations(use_case: str) -> str:
        """Get mode recommendation based on use case."""
        recommendations = {
            "development": "chain",
            "production": "graph",
            "debugging": "graph",
            "simple_qa": "chain",
            "complex_reasoning": "graph",
            "human_oversight": "graph",
            "batch_processing": "chain",
            "interactive": "graph"
        }
        
        return recommendations.get(use_case.lower(), "chain")


class AgentWrapper:
    """Wrapper to provide unified interface for both agent types."""
    
    def __init__(self, agent, mode: str):
        self.agent = agent
        self.mode = mode
        self._is_graph_agent = mode == ExecutionMode.GRAPH.value
    
    async def ask_question(self, question: str, **kwargs) -> str:
        """Ask a question using the appropriate agent method."""
        if self._is_graph_agent:
            thread_id = kwargs.get('thread_id', uuid.uuid4())
            return await self.agent.ask_question(question, thread_id)
        else:
            return await self.agent.ask_question(question)
    
    async def ask_question_stream(self, question: str, **kwargs):
        """Stream question response using the appropriate agent method."""
        if self._is_graph_agent:
            thread_id = kwargs.get('thread_id', 'default')
            async for chunk in self.agent.ask_question_stream(question, thread_id):
                yield chunk
        else:
            async for chunk in self.agent.ask_question_stream(question):
                yield chunk
    
    async def process_pdf(self, file_path, **kwargs) -> bool:
        """Process PDF file."""
        if hasattr(self.agent, 'process_pdf'):
            return await self.agent.process_pdf(file_path)
        else:
            # For graph agents, we might need to implement PDF processing
            logger.warning("PDF processing not implemented for graph agent")
            return False
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get knowledge base statistics."""
        if hasattr(self.agent, 'get_knowledge_base_stats'):
            return self.agent.get_knowledge_base_stats()
        elif hasattr(self.agent, 'vector_store'):
            return self.agent.vector_store.get_collection_stats()
        else:
            return {"document_count": 0, "collection_name": "Unknown"}
    
    def clear_knowledge_base(self) -> bool:
        """Clear knowledge base."""
        if hasattr(self.agent, 'clear_knowledge_base'):
            return self.agent.clear_knowledge_base()
        elif hasattr(self.agent, 'vector_store'):
            try:
                self.agent.vector_store.delete_collection()
                return True
            except Exception as e:
                logger.error(f"Error clearing knowledge base: {e}")
                return False
        else:
            return False
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get information about the current agent."""
        info = {
            "mode": self.mode,
            "type": "LangGraph" if self._is_graph_agent else "Chain",
            "features": []
        }
        
        if self._is_graph_agent:
            info["features"].extend([
                "State Management",
                "Checkpointing",
                "Error Recovery",
                "Workflow Observability"
            ])
            
            if hasattr(self.agent, 'enable_human_in_loop') and self.agent.enable_human_in_loop:
                info["features"].append("Human-in-the-Loop")
            
            if hasattr(self.agent, 'enable_summarization') and self.agent.enable_summarization:
                info["features"].append("Response Summarization")
        else:
            info["features"].extend([
                "Fast Execution",
                "Lightweight",
                "Simple Workflow"
            ])
        
        return info
    
    def get_workflow_state(self, thread_id: str = "default") -> Dict[str, Any]:
        """Get workflow state (graph agents only)."""
        if self._is_graph_agent and hasattr(self.agent, 'get_workflow_state'):
            return self.agent.get_workflow_state(thread_id)
        else:
            return {}
    
    def clear_workflow_state(self, thread_id: str = "default") -> bool:
        """Clear workflow state (graph agents only)."""
        if self._is_graph_agent and hasattr(self.agent, 'clear_workflow_state'):
            return self.agent.clear_workflow_state(thread_id)
        else:
            return True
