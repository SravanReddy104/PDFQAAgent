"""
Advanced retrieval service with multiple strategies.
Following Strategy Pattern: Different retrieval approaches.
"""
from typing import List, Dict, Any
from core.interfaces import RetrieverStrategy, VectorStore
from utils.logger import get_logger

logger = get_logger(__name__)


class BasicRetrieverStrategy(RetrieverStrategy):
    """Basic similarity search retrieval."""
    
    def retrieve(self, query: str, vector_store: VectorStore) -> List[Dict[str, Any]]:
        """Retrieve documents using basic similarity search."""
        try:
            results = vector_store.similarity_search(query, k=5)
            logger.info(f"Basic retrieval found {len(results)} documents")
            return results
        except Exception as e:
            logger.error(f"Error in basic retrieval: {e}")
            return []


class HybridRetrieverStrategy(RetrieverStrategy):
    """Hybrid retrieval combining multiple approaches."""
    
    def __init__(self, similarity_weight: float = 0.7, keyword_weight: float = 0.3):
        self.similarity_weight = similarity_weight
        self.keyword_weight = keyword_weight
    
    def retrieve(self, query: str, vector_store: VectorStore) -> List[Dict[str, Any]]:
        """Retrieve using hybrid approach."""
        try:
            # Get similarity-based results
            similarity_results = vector_store.similarity_search(query, k=8)

            logger.info("Similarity results: ", similarity_results)
            
            # Simple keyword-based filtering
            query_terms = set(query.lower().split())
            keyword_results = []
            
            for doc in similarity_results:
                content_terms = set(doc["content"].lower().split())
                keyword_overlap = len(query_terms.intersection(content_terms))
                
                # Calculate hybrid score
                similarity_score = doc.get("similarity_score", 0)
                keyword_score = keyword_overlap / len(query_terms) if query_terms else 0
                
                hybrid_score = (
                    self.similarity_weight * similarity_score +
                    self.keyword_weight * keyword_score
                )
                
                doc["hybrid_score"] = hybrid_score
                keyword_results.append(doc)
            
            # Sort by hybrid score and return top results
            keyword_results.sort(key=lambda x: x["hybrid_score"], reverse=True)
            results = keyword_results[:5]
            
            logger.info(f"Hybrid retrieval found {len(results)} documents")
            logger.info("Hybrid results: ", results)
            return results
            
        except Exception as e:
            logger.error(f"Error in hybrid retrieval: {e}")
            # Fallback to basic retrieval
            return BasicRetrieverStrategy().retrieve(query, vector_store)


class ContextualRetrieverStrategy(RetrieverStrategy):
    """Contextual retrieval with query expansion."""
    
    def retrieve(self, query: str, vector_store: VectorStore) -> List[Dict[str, Any]]:
        """Retrieve with contextual query expansion."""
        try:
            # Expand query with synonyms and related terms
            expanded_queries = self._expand_query(query)
            
            all_results = []
            for expanded_query in expanded_queries:
                results = vector_store.similarity_search(expanded_query, k=3)
                all_results.extend(results)
            
            # Remove duplicates and rank
            unique_results = self._deduplicate_results(all_results)
            ranked_results = self._rank_results(unique_results, query)
            
            logger.info(f"Contextual retrieval found {len(ranked_results)} documents")
            return ranked_results[:5]
            
        except Exception as e:
            logger.error(f"Error in contextual retrieval: {e}")
            return BasicRetrieverStrategy().retrieve(query, vector_store)
    
    def _expand_query(self, query: str) -> List[str]:
        """Expand query with related terms."""
        # Simple query expansion - could be enhanced with word embeddings
        base_query = query
        
        # Add variations
        expanded = [base_query]
        
        # Add question variations
        if not query.endswith('?'):
            expanded.append(f"{query}?")
        
        # Add "what is" variation
        if not query.lower().startswith(('what', 'how', 'why', 'when', 'where')):
            expanded.append(f"what is {query}")
        
        return expanded
    
    def _deduplicate_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate results based on content similarity."""
        unique_results = []
        seen_content = set()
        
        for result in results:
            content_hash = hash(result["content"][:100])  # Use first 100 chars as identifier
            if content_hash not in seen_content:
                seen_content.add(content_hash)
                unique_results.append(result)
        
        return unique_results
    
    def _rank_results(self, results: List[Dict[str, Any]], original_query: str) -> List[Dict[str, Any]]:
        """Rank results based on relevance to original query."""
        query_terms = set(original_query.lower().split())
        
        for result in results:
            content_terms = set(result["content"].lower().split())
            relevance_score = len(query_terms.intersection(content_terms)) / len(query_terms)
            result["relevance_score"] = relevance_score
        
        return sorted(results, key=lambda x: x.get("relevance_score", 0), reverse=True)


class RetrieverFactory:
    """Factory for creating retrieval strategies."""
    
    @staticmethod
    def create_retriever(strategy_type: str) -> RetrieverStrategy:
        """Create retriever strategy based on type."""
        strategies = {
            "basic": BasicRetrieverStrategy,
            "hybrid": HybridRetrieverStrategy,
            "contextual": ContextualRetrieverStrategy
        }
        
        if strategy_type not in strategies:
            logger.warning(f"Unknown retriever strategy {strategy_type}, using basic")
            strategy_type = "basic"
        
        return strategies[strategy_type]()
