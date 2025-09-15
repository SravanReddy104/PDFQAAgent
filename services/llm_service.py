"""
LLM service using Groq API.
Following Single Responsibility Principle: Handle only LLM operations.
"""
from typing import AsyncGenerator, List
import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from core.interfaces import LLMProvider
from config.settings import settings
from utils.logger import get_logger

logger = get_logger(__name__)


class GroqLLMService(LLMProvider):
    """Groq LLM service with streaming support."""
    
    def __init__(self):
        self.llm = ChatGroq(
            model=settings.groq_model,
            temperature=0.1,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )
        logger.info(f"Initialized Groq LLM with model: {settings.groq_model}")
    
    async def generate_response(self, prompt: str, context: str) -> AsyncGenerator[str, None]:
        """Generate streaming response from LLM."""
        try:
            system_prompt = self._create_system_prompt()
            user_prompt = self._create_user_prompt(prompt, context)
            
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            logger.info("Generated messages: ", messages)
            # Stream the response
            async for chunk in self.llm.astream(messages):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
                    
        except Exception as e:
            logger.error(f"Error generating LLM response: {e}")
            yield f"Error: {str(e)}"
    
    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for texts (not implemented for Groq)."""
        # Groq doesn't provide embeddings, this would use a separate service
        raise NotImplementedError("Groq doesn't provide embeddings. Use HuggingFace embeddings instead.")
    
    def _create_system_prompt(self) -> str:
        """Create system prompt for the LLM."""
        return """You are an intelligent PDF Q/A assistant. Your role is to answer questions based on the provided PDF document context.

Guidelines:
1. Answer questions accurately based ONLY on the provided context
2. If the answer is not in the context, clearly state that you don't have enough information
3. Provide detailed, well-structured answers when possible
4. Include relevant quotes from the source material when appropriate
5. If asked about specific pages or sections, reference them in your answer
6. Be concise but comprehensive in your responses
7. If the question is ambiguous, ask for clarification

Remember: Only use information from the provided context. Do not make up information or use external knowledge."""
    
    def _create_user_prompt(self, question: str, context: str) -> str:
        """Create user prompt with question and context."""
        return f"""Context from PDF documents:
{context}

Question: {question}

Please provide a detailed answer based on the context above."""
    
    async def generate_summary(self, text: str, max_length: int = 200) -> str:
        """Generate a summary of the provided text."""
        try:
            summary_prompt = f"""Please provide a concise summary of the following text in no more than {max_length} words:

{text}

Summary:"""
            
            messages = [HumanMessage(content=summary_prompt)]
            
            response = ""
            async for chunk in self.llm.astream(messages):
                if chunk.content:
                    response += chunk.content
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return "Error generating summary"
