"""
Web search service using LangChain's search tools.
Following Single Responsibility Principle: Handle only web search operations.
"""
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod
from config.settings import settings
from utils.logger import get_logger

# Initialize logger first
logger = get_logger(__name__)

# Web search imports
try:
    from langchain_tavily import TavilySearchResults
    TAVILY_AVAILABLE = True
    TAVILY_NEW_API = True
except ImportError:
    try:
        # Fallback to old import for backward compatibility
        from langchain_community.tools.tavily_search import TavilySearchResults
        TAVILY_AVAILABLE = True
        TAVILY_NEW_API = False
        logger.warning("Using deprecated TavilySearchResults. Please install: pip install -U langchain-tavily")
    except ImportError:
        TAVILY_AVAILABLE = False
        TAVILY_NEW_API = False

try:
    from langchain_community.tools import DuckDuckGoSearchResults
    DUCKDUCKGO_AVAILABLE = True
except ImportError as e:
    logger.warning(f"DuckDuckGo search not available: {e}")
    logger.info("Try installing with: pip install -U duckduckgo-search ddgs")
    DUCKDUCKGO_AVAILABLE = False


class WebSearchProvider(ABC):
    """Abstract base class for web search providers."""
    
    @abstractmethod
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search the web for the given query."""
        pass


class TavilySearchProvider(WebSearchProvider):
    """Tavily search provider - premium search with high quality results."""
    
    def __init__(self):
        if not TAVILY_AVAILABLE:
            raise ImportError("Tavily not available. Install with: pip install langchain-tavily")
        
        if not settings.tavily_api_key:
            raise ValueError("TAVILY_API_KEY is required for Tavily search")
        
        # Initialize with appropriate parameters based on API version
        if TAVILY_NEW_API:
            self.search_tool = TavilySearchResults(
                api_key=settings.tavily_api_key,
                max_results=settings.max_web_results
            )
        else:
            # Old API parameters
            self.search_tool = TavilySearchResults(
                api_key=settings.tavily_api_key,
                max_results=settings.max_web_results,
                search_depth="advanced",
                include_answer=True,
                include_raw_content=False
            )
        logger.info("Initialized Tavily search provider")
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search using Tavily API."""
        try:
            # Update max_results for this search
            self.search_tool.max_results = max_results
            
            results = self.search_tool.run(query)
            
            # Format results consistently
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append({
                        "title": result.get("title", ""),
                        "content": result.get("content", ""),
                        "url": result.get("url", ""),
                        "source": "tavily"
                    })
            
            logger.info(f"Tavily search found {len(formatted_results)} results for: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {e}")
            return []


class DuckDuckGoSearchProvider(WebSearchProvider):
    """DuckDuckGo search provider - free search with good privacy."""
    
    def __init__(self):
        if not DUCKDUCKGO_AVAILABLE:
            raise ImportError("DuckDuckGo search not available. Install with: pip install duckduckgo-search")
        
        self.search_tool = DuckDuckGoSearchResults(
            max_results=settings.max_web_results,
            region="wt-wt",  # Global
            safesearch="moderate"
        )
        logger.info("Initialized DuckDuckGo search provider")
    
    def search(self, query: str, max_results: int = 3) -> List[Dict[str, Any]]:
        """Search using DuckDuckGo."""
        try:
            # Update max_results for this search
            self.search_tool.max_results = max_results
            
            results = self.search_tool.run(query)
            
            # Format results consistently
            formatted_results = []
            if isinstance(results, list):
                for result in results:
                    if isinstance(result, dict):
                        formatted_results.append({
                            "title": result.get("title", ""),
                            "content": result.get("snippet", result.get("body", "")),
                            "url": result.get("link", result.get("url", "")),
                            "source": "duckduckgo"
                        })
            
            logger.info(f"DuckDuckGo search found {len(formatted_results)} results for: {query}")
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error in DuckDuckGo search: {e}")
            return []


class WebSearchService:
    """Main web search service following Strategy Pattern."""
    
    def __init__(self, provider: Optional[str] = None):
        self.provider_name = provider or settings.web_search_provider.lower()
        self.provider = self._create_provider()
    
    def _create_provider(self) -> Optional[WebSearchProvider]:
        """Create search provider based on configuration."""
        try:
            if self.provider_name == "tavily":
                return TavilySearchProvider()
            elif self.provider_name == "duckduckgo":
                return DuckDuckGoSearchProvider()
            else:
                logger.warning(f"Unknown search provider: {self.provider_name}")
                # Fallback to DuckDuckGo if available
                if DUCKDUCKGO_AVAILABLE:
                    return DuckDuckGoSearchProvider()
                return None
        except Exception as e:
            logger.error(f"Failed to initialize search provider {self.provider_name}: {e}")
            # Try fallback to DuckDuckGo
            try:
                if DUCKDUCKGO_AVAILABLE and self.provider_name != "duckduckgo":
                    logger.info("Falling back to DuckDuckGo search")
                    return DuckDuckGoSearchProvider()
            except Exception:
                pass
            return None
    
    def search(self, query: str, max_results: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search the web for information."""
        if not self.provider:
            logger.warning("No web search provider available")
            return []
        
        if not settings.enable_web_search:
            logger.info("Web search is disabled in settings")
            return []
        
        max_results = max_results or settings.max_web_results
        
        try:
            results = self.provider.search(query, max_results)
            logger.info(f"Web search completed: {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            return []
    
    def is_available(self) -> bool:
        """Check if web search is available and configured."""
        return self.provider is not None and settings.enable_web_search
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about the current search provider."""
        return {
            "provider": self.provider_name,
            "available": self.is_available(),
            "enabled": settings.enable_web_search,
            "max_results": settings.max_web_results
        }


class WebSearchFactory:
    """Factory for creating web search services."""
    
    @staticmethod
    def create_search_service(provider: Optional[str] = None) -> WebSearchService:
        """Create web search service with specified or configured provider."""
        return WebSearchService(provider)
    
    @staticmethod
    def get_available_providers() -> List[str]:
        """Get list of available search providers."""
        providers = []
        if TAVILY_AVAILABLE and settings.tavily_api_key:
            providers.append("tavily")
        if DUCKDUCKGO_AVAILABLE:
            providers.append("duckduckgo")
        return providers
