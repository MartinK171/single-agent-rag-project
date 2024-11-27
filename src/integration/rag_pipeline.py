from typing import Dict, Optional
import logging
import json
from langchain_ollama import OllamaLLM
from router.chain import RouterChain
from router.types import QueryType
from query_processing.processor import QueryProcessor
from query_processing.analyzer import QueryAnalysis
from vector_db.manager import VectorStoreManager
import re

class RAGPipeline:
    """Integrates routing, retrieval, and response generation."""

    def __init__(self, 
                 llm: Optional[OllamaLLM] = None,
                 vector_store_manager: Optional[VectorStoreManager] = None):
        """Initialize the RAG pipeline."""
        # Configure logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Initialize components
        self.llm = llm or OllamaLLM(
            model="llama2",
            host="ollama",
            port=11434
        )
        self.router = RouterChain(llm=self.llm)
        self.vector_store_manager = vector_store_manager
        self.query_processor = QueryProcessor()
        
        if not self.vector_store_manager:
            self.logger.warning("No vector store manager provided. Retrieval functionality will be limited.")

    def _augment_with_context(self, query: str, context: str) -> str:
        return f"""IMPORTANT: You are an AI assistant that MUST ONLY use information from the provided context.
    DO NOT use any other knowledge or make assumptions.
    If you cannot answer based solely on the context, say: "I cannot answer based on the provided context."

    Context:
    {context}

    Question: {query}

    Please provide your response based ONLY on the above context."""

    def _expand_query(self, query: str) -> str:
        """Expand query for better retrieval."""
        prompt = f"""Expand this query to improve search results:
Query: {query}
Expanded version:"""
        
        try:
            expanded = self.llm.invoke(prompt)
            self.logger.debug(f"Expanded query: {expanded}")
            return expanded.strip()
        except Exception as e:
            self.logger.error(f"Error expanding query: {str(e)}")
            return query

    def _clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        query = " ".join(query.split())
        query = query.lower()
        return query

    def _transform_query(self, query: str, query_type: QueryType) -> str:
        """Transform query based on type."""
        query = self._clean_query(query)
        if query_type == QueryType.RETRIEVAL:
            return self._expand_query(query)
        return query

    def _handle_retrieval(self, query: str, store_preference: Optional[str] = None) -> Dict:
        """Handle retrieval-type queries."""
        try:
            if not self.vector_store_manager:
                return self._handle_retrieval_fallback(query)
            
            # Determine best store if no preference given
            store_name = store_preference or self.vector_store_manager.determine_best_store(query)
            store = self.vector_store_manager.get_store(store_name)
            
            if not store:
                return self._handle_retrieval_fallback(query)
            
            # Search in selected store
            results = store.search(query, limit=3)
            
            # Log the search results for debugging
            self.logger.info(f"Search results for query '{query}': {results}")
            
            if not results:
                return self._handle_retrieval_fallback(query)
            
            # Extract the full text from each result
            context = "\n\n".join([r.get('text', '') for r in results])
            
            # Log the context being used
            self.logger.info(f"Using context for query '{query}': {context}")
            
            return {
                "context": context,
                "results": results,
                "selected_store": store_name,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Error in retrieval: {str(e)}")
            return self._handle_retrieval_fallback(query)

    def _handle_retrieval_fallback(self, query: str) -> Dict:
        """Fallback for when retrieval fails."""
        return {
            "context": "",
            "results": [],
            "success": False,
            "error": "Retrieval failed, no relevant information found."
        }

    def _handle_direct(self, query: str, template: str, analysis: QueryAnalysis) -> Dict:
        """Handle direct-type queries."""
        try:
            response = self.generate_response(query, '', template, analysis)
            return {
                "response": response.get("response", ""),
                "success": response.get("success", False),
                "query": query
            }
        except Exception as e:
            self.logger.error(f"Error in direct handling: {str(e)}")
            return {
                "response": "Failed to process direct query",
                "success": False,
                "error": str(e)
            }

    def _handle_calculation(self, query: str, template: str, analysis: QueryAnalysis) -> Dict:
        """Handle calculation-type queries."""
        try:
            response = self.generate_response(query, '', template, analysis)
            return {
                "response": response.get("response", ""),
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

    def process_query(self, query: str, store_preference: Optional[str] = None) -> Dict:
        """Process a query through the complete RAG pipeline."""
        try:
            # Analyze and process the query
            processed_result = self.query_processor.process(query)
            processed_query = processed_result.processed_query
            processing_path = processed_result.processing_path
            analysis = processed_result.analysis
            suggested_template = processed_result.suggested_template

            # Log processing info
            self.logger.info(f"Processing path: {processing_path}")
            self.logger.debug(f"Query analysis: {analysis}")

            # Route the query
            route_response = self.router.route(processed_query)
            query_type = route_response.query_type
            confidence = route_response.confidence

            # Handle based on query type
            if query_type == QueryType.RETRIEVAL:
                # Perform retrieval with store preference
                retrieval_result = self._handle_retrieval(processed_query, store_preference)
                response = self.generate_response(
                    processed_query,
                    retrieval_result.get('context', ''),
                    suggested_template,
                    analysis
                )
                response['selected_store'] = retrieval_result.get('selected_store')
                
            elif query_type == QueryType.DIRECT:
                response = self._handle_direct(processed_query, suggested_template, analysis)
                
            elif query_type == QueryType.CALCULATION:
                response = self._handle_calculation(processed_query, suggested_template, analysis)
                
            elif query_type == QueryType.CLARIFICATION:
                response = self._handle_clarification(processed_query)
                
            else:
                response = {"response": "Unsupported query type.", "success": False}

            # Record success
            self.query_processor.monitor.record_success(query, processed_result)

            # Build final result
            return {
                "query": query,
                "processed_query": processed_query,
                "query_type": query_type.value,
                "confidence": confidence,
                "response": response.get("response", ""),
                "success": response.get("success", False),
                "selected_store": response.get("selected_store"),
                "metadata": {
                    "processing_path": processing_path,
                    "analysis": analysis.__dict__,
                    "route_metadata": route_response.metadata,
                    "processing_metadata": processed_result.metadata
                }
            }

        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            self.query_processor.monitor.record_failure(query, str(e))
            return {
                "error": str(e),
                "query": query,
                "success": False,
                "response": "Failed to process query",
                "query_type": "",
                "confidence": 0.0,
            }

    def generate_response(self, query: str, context: str, template: str, analysis: QueryAnalysis) -> Dict:
        """Generate a response using LLM with context and template."""
        try:
            # Extract values for placeholders
            entity_details = ', '.join(analysis.entities) if analysis.entities else 'No specific entities identified.'
            complexity_analysis = f"Complexity score: {analysis.complexity}" if analysis.complexity > 0.7 else ''
            entities = ', '.join(analysis.entities) if analysis.entities else 'N/A'

            # Prepare prompt with stricter context usage
            prompt = f"""IMPORTANT: You are an AI assistant that MUST ONLY use information from the provided context.
DO NOT use any other knowledge. If you can't answer from the context, say so.

Context:
{context}

Question: {query}

Previous entities identified: {entity_details}

Instructions:
1. ONLY use facts stated in the context above
2. If you're unsure or the context doesn't contain the information, say "I cannot answer based on the provided context"
3. DO NOT make assumptions or use external knowledge
4. When you answer, include specific quotes from the context

Please provide your response:"""

            # Log the complete prompt for debugging
            self.logger.info(f"Complete prompt being sent to LLM: {prompt}")

            # Get LLM response
            response = self.llm.invoke(prompt)
            self.logger.debug(f"Raw LLM response: {response}")

            return {
                "response": response,
                "success": True,
                "context_used": context
            }

        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return {
                "response": "Failed to generate response",
                "error": str(e),
                "success": False
            }

    def get_default_template(self) -> str:
        """Get default template for response generation."""
        return """You are an assistant answering a question.

Question: {query}

Please provide a clear and concise response in JSON format:
{{"answer": "<your answer here>"}}
"""