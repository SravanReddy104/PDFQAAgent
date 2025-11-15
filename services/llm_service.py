"""
LLM service using Groq API and Hugging Face transformers.
Following Single Responsibility Principle: Handle only LLM operations.
"""
from typing import AsyncGenerator, List
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import PydanticOutputParser
from core.interfaces import LLMProvider
from config.settings import settings
from utils.logger import get_logger
from services.prompts import detectGreeting, systemPrompt, userPrompt
from services.models import GreetingClassifier

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

    async def call_llm(self, user_input: str) -> GreetingClassifier:
        """
        Classify if the input is a greeting or not.
        Returns a GreetingClassifier instance with 'result' field ('yes' or 'no').
        """
        try:
            prompt_template, parser = detectGreeting()
            chain = prompt_template | self.llm | parser
            # Pass the user's input to the chain
            response = await chain.ainvoke({"prompt": user_input})
            return response

        except Exception as e:
            logger.error(f"Error in call_llm: {e}")
            # Return a default response in case of error
            return GreetingClassifier(result="no")
    
    async def generate_response(self, prompt: str, context: str, conversation_history: List[str]):
        """Generate streaming response from LLM."""
        try:
            system_prompt = systemPrompt()
            user_prompt = userPrompt(prompt, context, conversation_history)
            
            messages = [
                SystemMessage(content=system_prompt.template),
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
