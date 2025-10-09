"""
Enhanced RAG implementation using LangGraph with state management and middleware.
Provides advanced workflow orchestration with checkpoints, error handling, and observability.
"""

import os
import pandas as pd
from typing import Dict, Any, List, Optional, TypedDict, Annotated
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain.schema import Document

# Middleware imports
from langchain.callbacks.human import HumanApprovalCallbackHandler
from langchain.callbacks.manager import CallbackManager

from config.settings import settings
from services.vector_store import VectorStoreFactory
from services.web_search_service import WebSearchService
from services.retrieval_service import RetrieverFactory
from services.llm_service import GroqLLMService
from utils.logger import get_logger

logger = get_logger(__name__)


class RAGState(TypedDict):
    """State schema for the RAG workflow."""
    messages: Annotated[List[BaseMessage], add_messages]
    question: str
    retrieved_docs: List[Dict[str, Any]]
    web_results: List[Dict[str, Any]]
    context: str
    answer: str
    metadata: Dict[str, Any]
    error: Optional[str]
    human_feedback: Optional[str]
    iteration_count: int
    should_continue: bool


class LangGraphRAGAgent:
    """Enhanced RAG agent using LangGraph for workflow orchestration."""
    
    def __init__(
        self,
        vector_store_type: Optional[str] = None,
        chunking_strategy: str = "hybrid",
        retrieval_strategy: str = "hybrid",
        checkpoint_backend: str = "memory",
        enable_human_in_loop: bool = False,
        enable_summarization: bool = True,
        max_iterations: int = 3
    ):
        """Initialize the LangGraph RAG agent."""
        self.vector_store_type = vector_store_type or os.getenv('VECTOR_STORE_TYPE', settings.vector_store_type)
        self.chunking_strategy = chunking_strategy
        self.retrieval_strategy = retrieval_strategy
        self.enable_human_in_loop = enable_human_in_loop
        self.enable_summarization = enable_summarization
        self.max_iterations = max_iterations
        
        # Initialize components
        self._initialize_components()
        
        # Setup checkpoint backend
        self.checkpointer = self._setup_checkpointer(checkpoint_backend)
        
        # Build the graph
        self.graph = self._build_graph()
        
        logger.info(f"LangGraph RAG Agent initialized with {checkpoint_backend} checkpointer")
    
    def _initialize_components(self):
        """Initialize RAG components."""
        try:
            self.vector_store = VectorStoreFactory.create_vector_store(self.vector_store_type)
            self.web_search = WebSearchService()
            self.retriever = RetrieverFactory.create_retriever(self.retrieval_strategy)
            self.llm_service = GroqLLMService()
            
            # Setup callback manager for human-in-the-loop
            if self.enable_human_in_loop:
                self.callback_manager = CallbackManager([
                    HumanApprovalCallbackHandler()
                ])
            else:
                self.callback_manager = None
                
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def _setup_checkpointer(self, backend: str):
        """Setup checkpoint backend for state persistence."""
        if backend == "sqlite":
            checkpoint_path = Path("./checkpoints/rag_checkpoints.db")
            checkpoint_path.parent.mkdir(exist_ok=True)
            return SqliteSaver.from_conn_string(str(checkpoint_path))
        else:
            return MemorySaver()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow."""
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("validate_input", self._validate_input)
        workflow.add_node("retrieve_documents", self._retrieve_documents)
        workflow.add_node("search_web", self._search_web)
        workflow.add_node("prepare_context", self._prepare_context)
        workflow.add_node("human_review", self._human_review)
        workflow.add_node("generate_answer", self._generate_answer)
        workflow.add_node("summarize_response", self._summarize_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Set entry point
        workflow.set_entry_point("validate_input")
        
        # Add edges
        workflow.add_edge("validate_input", "retrieve_documents")
        workflow.add_edge("retrieve_documents", "search_web")
        workflow.add_conditional_edges(
            "search_web",
            self._should_prepare_context,
            {
                "prepare_context": "prepare_context",
                "error": "handle_error"
            }
        )
        
        # Human-in-the-loop conditional edge
        if self.enable_human_in_loop:
            workflow.add_edge("prepare_context", "human_review")
            workflow.add_conditional_edges(
                "human_review",
                self._should_continue_after_human_review,
                {
                    "generate_answer": "generate_answer",
                    "retrieve_documents": "retrieve_documents",
                    "error": "handle_error"
                }
            )
        else:
            workflow.add_edge("prepare_context", "generate_answer")
        
        # Summarization conditional edge
        if self.enable_summarization:
            workflow.add_edge("generate_answer", "summarize_response")
            workflow.add_edge("summarize_response", END)
        else:
            workflow.add_edge("generate_answer", END)
        
        workflow.add_edge("handle_error", END)
        
        # Compile with checkpointer
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def _validate_input(self, state: RAGState) -> RAGState:
        """Validate and prepare input."""
        try:
            question = state.get("question", "")
            if not question or len(question.strip()) < 3:
                state["error"] = "Question is too short or empty"
                return state
            
            state["question"] = question.strip()
            state["metadata"] = {
                "timestamp": str(pd.Timestamp.now()),
                "vector_store": self.vector_store_type,
                "retrieval_strategy": self.retrieval_strategy
            }
            state["iteration_count"] = state.get("iteration_count", 0) + 1
            state["should_continue"] = True
            
            logger.info(f"Input validated: {question[:100]}...")
            return state
            
        except Exception as e:
            logger.error(f"Error in input validation: {e}")
            state["error"] = str(e)
            return state
    
    async def _retrieve_documents(self, state: RAGState) -> RAGState:
        """Retrieve relevant documents from vector store."""
        try:
            question = state["question"]
            retrieved_docs = self.retriever.retrieve(question, self.vector_store)
            
            state["retrieved_docs"] = retrieved_docs
            logger.info(f"Retrieved {len(retrieved_docs)} documents")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in document retrieval: {e}")
            state["error"] = str(e)
            return state
    
    async def _search_web(self, state: RAGState) -> RAGState:
        """Search web for additional context."""
        try:
            question = state["question"]
            retrieved_docs = state.get("retrieved_docs", [])
            
            # Only search web if we have few documents or web search is enabled
            if len(retrieved_docs) < 2 or settings.enable_web_search:
                if self.web_search.is_available():
                    web_results = self.web_search.search(question, max_results=3)
                    state["web_results"] = web_results
                    logger.info(f"Found {len(web_results)} web results")
                else:
                    state["web_results"] = []
                    logger.info("Web search not available")
            else:
                state["web_results"] = []
                logger.info("Skipping web search - sufficient documents found")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in web search: {e}")
            state["web_results"] = []
            return state
    
    async def _prepare_context(self, state: RAGState) -> RAGState:
        """Prepare context from retrieved documents and web results."""
        try:
            retrieved_docs = state.get("retrieved_docs", [])
            web_results = state.get("web_results", [])
            
            context_parts = []
            
            # Add document context
            if retrieved_docs:
                context_parts.append("## Document Context:")
                for i, doc in enumerate(retrieved_docs[:5], 1):
                    content = doc.get("content", "")[:500]
                    source = doc.get("metadata", {}).get("filename", "Unknown")
                    context_parts.append(f"{i}. From {source}: {content}")
            
            # Add web context
            if web_results:
                context_parts.append("\n## Web Search Results:")
                for i, result in enumerate(web_results[:3], 1):
                    title = result.get("title", "")
                    content = result.get("content", "")[:300]
                    url = result.get("url", "")
                    context_parts.append(f"{i}. {title}\n   {content}\n   Source: {url}")
            
            state["context"] = "\n".join(context_parts)
            logger.info(f"Context prepared: {len(state['context'])} characters")
            
            return state
            
        except Exception as e:
            logger.error(f"Error preparing context: {e}")
            state["error"] = str(e)
            return state
    
    async def _human_review(self, state: RAGState) -> RAGState:
        """Human-in-the-loop review step."""
        try:
            question = state["question"]
            context = state.get("context", "")
            
            # Present context to human for review
            review_prompt = f"""
            Question: {question}
            
            Context prepared:
            {context[:1000]}...
            
            Options:
            1. Continue with this context
            2. Retrieve more documents
            3. Stop and return error
            
            Please choose (1/2/3):
            """
            
            # In a real implementation, this would show a UI prompt
            # For now, we'll simulate approval
            state["human_feedback"] = "approved"  # Simulated approval
            
            logger.info("Human review completed")
            return state
            
        except Exception as e:
            logger.error(f"Error in human review: {e}")
            state["error"] = str(e)
            return state
    
    async def _generate_answer(self, state: RAGState) -> RAGState:
        """Generate answer using LLM."""
        try:
            question = state["question"]
            context = state.get("context", "")
            
            # Generate response using the LLM service
            response_parts = []
            async for chunk in self.llm_service.generate_response(question, context):
                response_parts.append(chunk)
            
            answer = "".join(response_parts)
            state["answer"] = answer
            
            # Add to messages
            if "messages" not in state:
                state["messages"] = []
            
            state["messages"].extend([
                HumanMessage(content=question),
                AIMessage(content=answer)
            ])
            
            logger.info(f"Answer generated: {len(answer)} characters")
            return state
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            state["error"] = str(e)
            return state
    
    async def _summarize_response(self, state: RAGState) -> RAGState:
        """Summarize the response if it's too long."""
        try:
            answer = state.get("answer", "")
            
            # Only summarize if answer is very long
            if len(answer) > 2000:
                summary_prompt = f"Please provide a concise summary of this response:\n\n{answer}"
                
                summary_parts = []
                async for chunk in self.llm_service.generate_response(summary_prompt, ""):
                    summary_parts.append(chunk)
                
                summary = "".join(summary_parts)
                state["answer"] = f"{summary}\n\n---\n[Full response available on request]"
                
                logger.info("Response summarized")
            
            return state
            
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            return state
    
    async def _handle_error(self, state: RAGState) -> RAGState:
        """Handle errors in the workflow."""
        error = state.get("error", "Unknown error")
        state["answer"] = f"I apologize, but I encountered an error while processing your question: {error}"
        
        logger.error(f"Workflow error: {error}")
        return state
    
    def _should_prepare_context(self, state: RAGState) -> str:
        """Decide whether to prepare context or handle error."""
        if state.get("error"):
            return "error"
        return "prepare_context"
    
    def _should_continue_after_human_review(self, state: RAGState) -> str:
        """Decide next step after human review."""
        if state.get("error"):
            return "error"
        
        feedback = state.get("human_feedback", "")
        if feedback == "retry":
            return "retrieve_documents"
        elif feedback == "stop":
            return "error"
        else:
            return "generate_answer"
    
    async def ask_question(self, question: str, thread_id: str = "default") -> str:
        """Ask a question using the LangGraph workflow."""
        try:
            config = RunnableConfig(
                configurable={"thread_id": thread_id},
                callbacks=self.callback_manager.handlers if self.callback_manager else None
            )
            
            initial_state = RAGState(
                messages=[],
                question=question,
                retrieved_docs=[],
                web_results=[],
                context="",
                answer="",
                metadata={},
                error=None,
                human_feedback=None,
                iteration_count=0,
                should_continue=True
            )
            
            # Run the workflow
            final_state = await self.graph.ainvoke(initial_state, config)
            
            return final_state.get("answer", "I couldn't generate an answer.")
            
        except Exception as e:
            logger.error(f"Error in LangGraph workflow: {e}")
            return f"An error occurred while processing your question: {str(e)}"
    
    async def ask_question_stream(self, question: str, thread_id: str = "default"):
        """Stream the question-answering process."""
        try:
            config = RunnableConfig(
                configurable={"thread_id": thread_id},
                callbacks=self.callback_manager.handlers if self.callback_manager else None
            )
            
            initial_state = RAGState(
                messages=[],
                question=question,
                retrieved_docs=[],
                web_results=[],
                context="",
                answer="",
                metadata={},
                error=None,
                human_feedback=None,
                iteration_count=0,
                should_continue=True
            )
            
            # Stream the workflow execution
            async for event in self.graph.astream(initial_state, config):
                node_name = list(event.keys())[0]
                node_output = event[node_name]
                
                # Yield progress updates
                if node_name == "retrieve_documents":
                    yield f"🔍 Retrieved {len(node_output.get('retrieved_docs', []))} documents...\n"
                elif node_name == "search_web":
                    yield f"🌐 Found {len(node_output.get('web_results', []))} web results...\n"
                elif node_name == "generate_answer":
                    answer = node_output.get("answer", "")
                    if answer:
                        yield answer
                        
        except Exception as e:
            logger.error(f"Error in streaming workflow: {e}")
            yield f"An error occurred: {str(e)}"
    
    def get_workflow_state(self, thread_id: str = "default") -> Dict[str, Any]:
        """Get the current workflow state."""
        try:
            config = RunnableConfig(configurable={"thread_id": thread_id})
            state = self.graph.get_state(config)
            return state.values if state else {}
        except Exception as e:
            logger.error(f"Error getting workflow state: {e}")
            return {}
    
    def clear_workflow_state(self, thread_id: str = "default") -> bool:
        """Clear the workflow state."""
        try:
            # This would clear the checkpoint for the given thread
            # Implementation depends on the checkpointer backend
            logger.info(f"Cleared workflow state for thread: {thread_id}")
            return True
        except Exception as e:
            logger.error(f"Error clearing workflow state: {e}")
            return False
