from typing import Dict, Optional, List
import logging
import time
from langchain_ollama import OllamaLLM
from router.chain import RouterChain
from router.types import QueryType
from query_processing.processor import QueryProcessor
from query_processing.analyzer import QueryAnalysis
from vector_db.manager import VectorStoreManager
from tools.web_search import WebSearchTool
from tools.calculator import Calculator

class RAGPipeline:
    """Integrates routing, retrieval, and response generation with multiple tools."""

    def __init__(self, 
                 llm: Optional[OllamaLLM] = None,
                 vector_store_manager: Optional[VectorStoreManager] = None):
        """Initialize the RAG pipeline with necessary components."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        self.llm = llm or OllamaLLM(
            model="llama2",
            host="ollama",
            port=11434
        )
        self.router = RouterChain(llm=self.llm)
        self.vector_store_manager = vector_store_manager
        self.query_processor = QueryProcessor()
        
        # Initialize tools
        self.web_search_tool = WebSearchTool(
            max_retries=3,
            base_delay=2.0
        )
        self.calculator = Calculator()

    def _select_tool(self, query_type: QueryType, store_preference: Optional[str] = None):
        """Select appropriate tool based on query type."""
        tool_mapping = {
            QueryType.WEB_SEARCH: self.web_search_tool,
            QueryType.CALCULATION: self.calculator,
            QueryType.RETRIEVAL: self.vector_store_manager,
            QueryType.DIRECT: None
        }
        return tool_mapping.get(query_type)

    def _augment_with_context(self, query: str, context: str) -> str:
        """Create prompt with context for retrieval queries."""
        return f"""IMPORTANT: You are an AI assistant that MUST ONLY use information from the provided context.
DO NOT use any other knowledge. If you can't answer from the context, say so.

Context:
{context}

Question: {query}

Instructions:
1. ONLY use facts stated in the context above
2. If you're unsure or the context doesn't contain the information, say "I cannot answer based on the provided context"
3. DO NOT make assumptions or use external knowledge
4. When you answer, include specific quotes from the context

Please provide your response:"""

    def _augment_with_context_for_web_search(self, query: str, context: str) -> str:
        """Create specialized prompt for web search results."""
        return f"""You are an AI assistant helping with current information.
Please analyze these recent search results and provide a comprehensive answer.

Search Results:
{context}

Question: {query}

Instructions:
1. Synthesize information from the search results
2. Maintain factual accuracy
3. Acknowledge if information is limited or unclear
4. Include relevant dates when available
5. Be clear about the recency of the information
6. If the results don't contain enough information, say so

Please provide your response:"""

    def _format_search_results(self, results: List[Dict]) -> str:
        """Format search results into a coherent context."""
        formatted = []
        for i, result in enumerate(results, 1):
            formatted.extend([
                f"[Source {i}]",
                f"Title: {result['title']}",
                f"Content: {result['snippet']}",
                f"URL: {result.get('link', 'Not available')}",
                "---"
            ])
        return "\n".join(formatted)

    def _handle_web_search(self, query: str) -> Dict:
        """Handle web search queries with improved response generation."""
        try:
            self.logger.info(f"Performing web search for query: {query}")
            
            # Get search results
            results = self.web_search_tool.search(query)
            self.logger.info(f"Retrieved {len(results)} search results")
            
            if not results:
                self.logger.warning("No search results found")
                return {
                    "success": False,
                    "error": "No search results found",
                    "response": (
                        "I apologize, but I couldn't find any recent information about that topic. "
                        "You could try:\n"
                        "1. Rephrasing your question\n"
                        "2. Being more specific about the timeframe\n"
                        "3. Checking news websites directly"
                    ),
                    "fallback_used": True
                }
            
            # Format context from results
            context = self.web_search_tool.format_results(results)
            
            # Create a specialized prompt for current information
            prompt = f"""Based on the following recent search results, provide a comprehensive answer about {query}

    Search Results:
    {context}

    Required Format:
    1. Start with a brief overview
    2. List the major developments chronologically if possible
    3. Include dates when available
    4. Cite sources when stating specific facts
    5. If any information seems outdated or unclear, acknowledge this

    Additional Instructions:
    - Only use information from the provided search results
    - Acknowledge if the results don't provide enough information
    - Don't make assumptions about dates or details not in the results
    - If the results seem outdated, mention this
    - Be clear about which source each piece of information comes from

    Your response:"""

            # Get LLM response
            response = self.llm.invoke(prompt)
            
            return {
                "success": True,
                "response": response,
                "context": context,
                "metadata": {
                    "result_count": len(results),
                    "sources": [r.get('link') for r in results],
                    "search_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "sources_used": [r.get('source') for r in results]
                }
            }
            
        except Exception as e:
            self.logger.error(f"Web search error: {str(e)}", exc_info=True)
            return {
                "success": False,
                "error": str(e),
                "response": "I apologize, but I'm having trouble accessing current information right now.",
                "fallback_used": True
            }

    def _handle_retrieval(self, query: str, store_preference: Optional[str] = None) -> Dict:
        """Handle retrieval from vector store."""
        try:
            if not self.vector_store_manager:
                return self._handle_retrieval_fallback(query)
            
            store_name = store_preference or self.vector_store_manager.determine_best_store(query)
            store = self.vector_store_manager.get_store(store_name)
            
            if not store:
                return self._handle_retrieval_fallback(query)
            
            results = store.search(query, limit=3)
            self.logger.info(f"Search results for query '{query}': {results}")
            
            if not results:
                return self._handle_retrieval_fallback(query)
            
            context = "\n\n".join([r.get('text', '') for r in results])
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

    def _handle_calculation(self, query: str) -> Dict:
        """Handle calculation queries."""
        try:
            expression = self._extract_math_expression(query)
            if not expression:
                return {"success": False, "error": "No valid mathematical expression found"}
            
            result = self.calculator.evaluate(expression)
            return {
                "result": result,
                "expression": expression,
                "success": True
            }
        except Exception as e:
            self.logger.error(f"Error in calculation: {str(e)}")
            return {"success": False, "error": str(e)}

    def _handle_direct(self, query: str) -> Dict:
        """Handle direct queries that don't need retrieval."""
        try:
            prompt = f"""Answer this question directly and concisely: {query}
            
Instructions:
1. If this is a request for a joke, provide a family-friendly joke
2. If this is a general question, provide a clear, direct answer
3. If you're not sure, say so
4. Be concise but informative"""
            
            response = self.llm.invoke(prompt)
            
            return {
                "response": response,
                "success": True,
                "metadata": {"direct_response": True}
            }
        except Exception as e:
            self.logger.error(f"Error in direct response: {str(e)}")
            return {
                "response": "I apologize, but I'm having trouble processing that request at the moment.",
                "success": False,
                "error": str(e)
            }

    def _handle_search_failure(self, query: str) -> Dict:
        """Handle cases where web search fails."""
        try:
            self.logger.info("Attempting fallback to direct query handling")
            direct_result = self._handle_direct(query)
            if direct_result.get("success", False):
                direct_result["fallback_used"] = True
                return direct_result
        except Exception as e:
            self.logger.error(f"Fallback handling failed: {str(e)}")
        
        return {
            "success": False,
            "response": (
                "I apologize, but I'm currently unable to access real-time information, "
                "and this question requires current data to answer accurately. "
                "Please try again later or rephrase your question."
            ),
            "error": "Search failed and fallback unavailable",
            "fallback_used": True
        }

    def _handle_retrieval_fallback(self, query: str) -> Dict:
        """Handle retrieval failures."""
        return {
            "context": "",
            "results": [],
            "success": False,
            "error": "Retrieval failed, no relevant information found."
        }

    def _extract_math_expression(self, query: str) -> Optional[str]:
        """Extract mathematical expression from query."""
        import re
        math_pattern = r'(\d+[\s\+\-\*/\(\)]+\d+)'
        match = re.search(math_pattern, query)
        return match.group(1) if match else None

    def generate_response(self, query: str, context: str, template: str, analysis: QueryAnalysis) -> Dict:
        """Generate response using context and template."""
        try:
            prompt = self._augment_with_context(query, context)
            self.logger.info(f"Complete prompt being sent to LLM: {prompt}")
            
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

    def process_query(self, query: str, store_preference: Optional[str] = None) -> Dict:
        """Process a query through the pipeline."""
        try:
            self.logger.info(f"Processing query: {query}")
            
            # Process and route query
            processed_result = self.query_processor.process(query)
            processed_query = processed_result.processed_query
            route_response = self.router.route(processed_query)
            
            query_type = route_response.query_type
            confidence = route_response.confidence

            # Handle different query types
            if query_type == QueryType.WEB_SEARCH:
                self.logger.info("Processing web search query")
                web_result = self._handle_web_search(processed_query)
                
                if not web_result.get("success"):
                    self.logger.info("Web search failed, attempting fallback")
                    fallback_result = self._handle_search_failure(processed_query)
                    response = {
                        "response": fallback_result.get("response", ""),
                        "success": fallback_result.get("success", False),
                        "metadata": fallback_result.get("metadata", {}),
                        "fallback_used": True
                    }
                else:
                    response = {
                        "response": web_result.get("response", ""),
                        "success": web_result.get("success", True),
                        "metadata": {
                            **web_result.get("metadata", {}),
                            "web_search": True
                        }
                    }
                
            elif query_type == QueryType.DIRECT:
                direct_result = self._handle_direct(processed_query)
                response = {
                    "response": direct_result.get("response", ""),
                    "success": direct_result.get("success", False),
                    "metadata": direct_result.get("metadata", {})
                }
                
            elif query_type == QueryType.CALCULATION:
                calc_result = self._handle_calculation(processed_query)
                response = {
                    "response": f"The result is {calc_result.get('result')}",
                    "success": calc_result.get("success", False),
                    "metadata": {"expression": calc_result.get("expression")}
                }
                
            elif query_type == QueryType.RETRIEVAL:
                retrieval_result = self._handle_retrieval(processed_query, store_preference)
                if retrieval_result.get("success"):
                    response = self.generate_response(
                        processed_query,
                        retrieval_result.get('context', ''),
                        processed_result.suggested_template,
                        processed_result.analysis
                    )
                    response['selected_store'] = retrieval_result.get('selected_store')
                else:
                    response = {
                        "response": "Could not find relevant information in the documents.",
                        "success": False,
                        "fallback_used": True
                    }
            
            else:
                response = {
                    "response": "Unsupported query type.",
                    "success": False
                }

            # Record success and return result
            self.query_processor.monitor.record_success(query, processed_result)

            return {
                "query": query,
                "processed_query": processed_query,
                "query_type": query_type.value,
                "confidence": confidence,
                "response": response.get("response", ""),
                "success": response.get("success", False),
                "selected_store": response.get("selected_store"),
                "metadata": {
                    "processing_path": processed_result.processing_path,
                    "analysis": processed_result.analysis.__dict__,
                    "route_metadata": route_response.metadata,
                    "processing_metadata": processed_result.metadata,
                    "fallback_used": response.get("fallback_used", False)
                }
            }

        except Exception as e:
            self.logger.error(f"Pipeline error: {str(e)}")
            self.query_processor.monitor.record_failure(query, str(e))
            return {
                "query": query,
                "response": "I apologize, but I encountered an error processing your request.",
                "query_type": "",
                "confidence": 0.0,
                "success": False,
                "metadata": {"error": str(e)}
            }