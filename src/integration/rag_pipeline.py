from typing import Dict, Optional
import logging
import json
from langchain_ollama import OllamaLLM
from src.router.chain import RouterChain
from src.router.types import QueryType
from src.vector_db.store import VectorStore
from query_processing.processor import QueryProcessor
from query_processing.analyzer import QueryAnalysis  # Import added here

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
        self.query_processor = QueryProcessor()
        
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

    def generate_response(self, query: str, context: str, template: str, analysis: QueryAnalysis) -> Dict:
        """Generate a response using LLM with context and template."""
        try:
            # Extract values for placeholders
            entity_details = ', '.join(analysis.entities) if analysis.entities else 'No specific entities identified.'
            complexity_analysis = f"Complexity score: {analysis.complexity}" if analysis.complexity > 0.7 else ''
            entities = ', '.join(analysis.entities) if analysis.entities else 'N/A'

            # Prepare the prompt using the template
            prompt = template.format(
                query=query,
                context=context,
                entity_details=entity_details,
                complexity_analysis=complexity_analysis,
                entities=entities
            )

            # Get LLM response
            response = self.llm.invoke(prompt)

            # Log the LLM response
            self.logger.debug(f"LLM response: {response}")

            # Parse the response as JSON
            response_data = json.loads(response)

            # Extract the answer
            answer = response_data.get("answer", "")

            return {
                "response": answer,
                "success": True,
                "context_used": context
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse LLM response as JSON: {str(e)}")
            return {
                "response": "Failed to generate response",
                "error": "Invalid response format from LLM",
                "success": False
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
            return self._handle_direct(query, template=self.get_default_template(), analysis=QueryAnalysis())
        except Exception as e:
            self.logger.error(f"Error in retrieval fallback: {str(e)}")
            return {
                "response": "Unable to process query",
                "success": False,
                "error": str(e)
            }

    def _handle_direct(self, query: str, template: str, analysis: QueryAnalysis) -> Dict:
        """Handle direct-type queries."""
        try:
            response = self.generate_response(query, context='', template=template, analysis=analysis)
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
            response = self.generate_response(query, context='', template=template, analysis=analysis)
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
            # Analyze and process the query
            processed_result = self.query_processor.process(query)
            processed_query = processed_result.processed_query
            processing_path = processed_result.processing_path
            analysis = processed_result.analysis
            suggested_template = processed_result.suggested_template

            # Log the processing path and analysis
            self.logger.info(f"Processing path: {processing_path}")
            self.logger.debug(f"Query analysis: {analysis}")

            # Use the processed query for routing
            route_response = self.router.route(processed_query)
            query_type = route_response.query_type
            confidence = route_response.confidence

            # Handle based on query type
            if query_type == QueryType.RETRIEVAL:
                # Perform retrieval and generate response
                retrieval_result = self._handle_retrieval(processed_query)
                response = self.generate_response(
                    processed_query,
                    retrieval_result.get('context', ''),
                    suggested_template,
                    analysis
                )
            elif query_type == QueryType.DIRECT:
                # Directly generate response without retrieval
                response = self._handle_direct(processed_query, suggested_template, analysis)
            elif query_type == QueryType.CALCULATION:
                # Handle calculation queries
                response = self._handle_calculation(processed_query, suggested_template, analysis)
            elif query_type == QueryType.CLARIFICATION:
                # Ask for clarification
                response = self._handle_clarification(processed_query)
            else:
                response = {"response": "Unsupported query type.", "success": False}

            # Record success in the monitor
            self.query_processor.monitor.record_success(query, processed_result)

            # Build the final result
            result = {
                "query": query,
                "processed_query": processed_query,
                "query_type": query_type.value,
                "confidence": confidence,
                "response": response.get("response", ""),
                "success": response.get("success", False),
                "metadata": {
                    "processing_path": processing_path,
                    "analysis": analysis,
                    "route_metadata": route_response.metadata,
                    "processing_metadata": processed_result.metadata
                }
            }

            return result

        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            # Record failure in the monitor
            self.query_processor.monitor.record_failure(query, str(e))
            return {
                "error": str(e),
                "query": query,
                "success": False,
                "response": "Failed to process query",
                "query_type": "",
                "confidence": 0.0,
            }

    def get_default_template(self) -> str:
        """Get a default template for generating responses."""
        # Return a basic template string
        return """You are an assistant.

{query}

Please provide an answer in JSON format:

{{
    "answer": "<Your answer here>"
}}
"""

