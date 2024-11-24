from typing import Dict, Optional
import logging
from langchain_ollama import OllamaLLM
from src.router.chain import RouterChain
from src.router.types import QueryType
from src.vector_db.store import VectorStore

class RAGPipeline:
    """Integrates routing, retrieval, and response generation."""
    
    def __init__(self, 
                llm: Optional[OllamaLLM] = None,
                vector_store: Optional[VectorStore] = None):
        """Initialize the RAG pipeline."""
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.llm = llm or OllamaLLM(
            model="llama2",
            host="ollama",
            port=11434
        )
        self.router = RouterChain(llm=self.llm)
        self.vector_store = vector_store
        
        if not self.vector_store:
            self.logger.warning("No vector store provided. Retrieval functionality will be limited.")

    def _augment_with_context(self, query: str, context: str) -> str:
        """Augment LLM prompt with retrieved context."""
        return f"""Use the following context to help answer the question.

Context:
{context}

Question: {query}

Please provide a response that:
1. Answers the question using the context
2. Cites specific parts of the context
3. Says "I don't know" if the context doesn't contain the answer
"""

    def _expand_query(self, query: str) -> str:
        """Expand query for better retrieval."""
        prompt = f"""Expand this query to improve search results:
Query: {query}
Expanded version:"""
        
        try:
            expanded = self.llm.invoke(prompt)
            return expanded.strip()
        except Exception as e:
            self.logger.error(f"Error expanding query: {str(e)}")
            return query

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove extra whitespace
        query = " ".join(query.split())
        # Convert to lowercase
        query = query.lower()
        return query

    def _transform_query(self, query: str, query_type: QueryType) -> str:
        """Transform query based on type."""
        query = self._clean_query(query)
        if query_type == QueryType.RETRIEVAL:
            return self._expand_query(query)
        return query

    def generate_response(self, query: str, context: str) -> Dict:
        """Generate a response using LLM with context."""
        try:
            # Augment query with context
            prompt = self._augment_with_context(query, context)
            
            # Get LLM response
            response = self.llm.invoke(prompt)
            
            return {
                "response": response,
                "context_used": context,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "Failed to generate response",
                "error": str(e),
                "success": False
            }

    def _handle_retrieval(self, query: str, retrieval_query: Optional[str] = None) -> Dict:
        """Handle retrieval-type queries."""
        try:
            if not self.vector_store:
                return self._handle_retrieval_fallback(query)
            
            search_query = retrieval_query or query
            results = self.vector_store.search(search_query, limit=3)
            
            if not results:
                return self._handle_retrieval_fallback(query)
            
            context = "\n".join([r.text for r in results])
            
            return {
                "context": context,
                "results": results,
                "query_used": search_query,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in retrieval: {str(e)}")
            return self._handle_retrieval_fallback(query)

    def _handle_retrieval_fallback(self, query: str) -> Dict:
        """Fallback for when retrieval fails."""
        try:
            # Try direct response without context
            return self._handle_direct(query)
        except Exception as e:
            self.logger.error(f"Error in retrieval fallback: {str(e)}")
            return {
                "response": "Unable to process query",
                "success": False,
                "error": str(e)
            }

    def _handle_direct(self, query: str) -> Dict:
        """Handle direct-type queries."""
        try:
            response = self.llm.invoke(query)
            return {
                "response": response,
                "success": True,
                "query": query
            }
        except Exception as e:
            self.logger.error(f"Error in direct handling: {str(e)}")
            return {
                "response": "Failed to process direct query",
                "success": False,
                "error": str(e)
            }

    def _handle_calculation(self, query: str) -> Dict:
        """Handle calculation-type queries."""
        try:
            # For calculation queries, we might want to format them specifically
            prompt = f"Please calculate: {query}"
            response = self.llm.invoke(prompt)
            
            return {
                "response": response,
                "success": True,
                "query": query
            }
        except Exception as e:
            self.logger.error(f"Error in calculation: {str(e)}")
            return {
                "response": "Failed to process calculation",
                "success": False,
                "error": str(e)
            }

    def _handle_clarification(self, query: str) -> Dict:
        """Handle queries needing clarification."""
        return {
            "response": "I need more information. Could you please provide more context about what you're asking?",
            "query": query,
            "clarification_needed": True,
            "success": True
        }

    def _handle_with_retry(self, handler_func, query: str, max_retries: int = 3) -> Dict:
        """Execute handler with retries."""
        for attempt in range(max_retries):
            try:
                return handler_func(query)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(f"All retry attempts failed: {str(e)}")
                    return {
                        "error": str(e),
                        "success": False,
                        "response": "Failed after retries"
                    }
                self.logger.warning(f"Retry attempt {attempt + 1} failed: {str(e)}")

    def process_query(self, query: str) -> Dict:
        """Process a query through the complete RAG pipeline."""
        try:
            # Route the query
            route_response = self.router.route(query)
            
            # Transform query if needed
            transformed_query = self._transform_query(
                query, 
                route_response.query_type
            )
            
            # Initialize the response dictionary
            response = {}
            
            # Process based on query type
            if route_response.query_type == QueryType.RETRIEVAL:
                # Handle retrieval-type queries
                retrieval_result = self._handle_retrieval(
                    transformed_query,
                    route_response.retrieval_query
                )
                
                if retrieval_result.get("context"):
                    # If context is available, generate a response using it
                    response = self.generate_response(
                        query,
                        retrieval_result["context"]
                    )
                else:
                    # If no context, determine if an error occurred
                    if "error" in retrieval_result:
                        response = {
                            "response": "Failed to process direct query",
                            "success": False,
                            "error": retrieval_result["error"]
                        }
                    else:
                        response = {
                            "response": "No relevant information found",
                            "success": False
                        }
                        
            elif route_response.query_type == QueryType.CALCULATION:
                # Handle calculation-type queries
                response = self._handle_calculation(transformed_query)
            elif route_response.query_type == QueryType.CLARIFICATION:
                # Handle clarification-type queries
                response = self._handle_clarification(transformed_query)
            else:  # DIRECT
                # Handle direct-type queries
                response = self._handle_direct(transformed_query)
            
            # Build the combined result
            result = {
                "query": query,
                "transformed_query": transformed_query,
                "query_type": route_response.query_type.value,
                "confidence": route_response.confidence,
                "response": response.get("response", ""),
                "context_used": response.get("context_used", ""),
                "success": response.get("success", False),
                "metadata": route_response.metadata
            }
            
            # Merge any additional keys from the handler response
            result.update(response)
            
            return result
                
        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            return {
                "error": str(e),
                "query": query,
                "success": False,
                "response": "Failed to process query"
            }
