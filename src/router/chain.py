from typing import Dict, Optional
import logging
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from .types import QueryType, RouterResponse
from .prompts import ROUTER_PROMPT_TEMPLATE
import json

class RouterChain:
    """Chain for routing queries to appropriate handlers."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """
        Initialize the RouterChain with an LLM and a prompt template.

        Args:
            llm: An instance of OllamaLLM, or None to use default configuration.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        self.llm = llm or OllamaLLM(
            model="llama2",
            host="ollama", 
            port=11434
        )
        
        self.prompt = PromptTemplate(
            template=ROUTER_PROMPT_TEMPLATE,
            input_variables=["query"]
        )
    
    def _parse_response(self, response: str) -> Dict:
        """Parse LLM response using JSON parsing."""
        self.logger.debug(f"Parsing response: {response}")
        try:
            parsed = json.loads(response)
            self.logger.debug(f"Parsed response: {parsed}")

            # Ensure all expected fields are present
            return {
                "query_type": parsed.get("query_type", "retrieval").lower(),
                "confidence": float(parsed.get("confidence", 0.5)),
                "should_retrieve": parsed.get("should_retrieve", True),
                "retrieval_query": parsed.get("retrieval_query", ""),
                "reasoning": parsed.get("reasoning", "No reasoning provided."),
            }
        except json.JSONDecodeError as e:
            self.logger.error(f"Error parsing JSON response: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing response: {e}")
        
        return {
            "query_type": "retrieval",
            "confidence": 0.5,
            "should_retrieve": True,
            "retrieval_query": "",
            "reasoning": "Failed to parse response.",
        }

    def _check_query_patterns(self, query: str) -> Dict[str, bool]:
        """Check query for specific patterns."""
        query_lower = query.lower()
        return {
            'is_web_search': any(word in query_lower for word in [
                'latest', 'recent', 'current', 'now', 'today', 'news', 
                'update', 'newest', 'this week', 'this month'
            ]),
            'is_calculation': any(c in query for c in ['+', '-', '*', '/', '%']) 
                            or any(word in query_lower for word in ['calculate', 'compute', 'sum']),
            'is_direct': any(word in query_lower for word in ['what is', 'who is', 'tell me', 'explain'])
                        and len(query.split()) < 6,
            'is_retrieval': any(word in query_lower for word in ['document', 'docs', 'content', 'find in'])
        }
    
    def route(self, query: str) -> RouterResponse:
        """
        Route the query to the appropriate handler.
        
        Args:
            query: The user's query string
            
        Returns:
            RouterResponse containing routing information
        """
        try:
            # Get initial LLM routing suggestion
            formatted_prompt = self.prompt.format(query=query)
            response = self.llm.invoke(formatted_prompt)
            parsed = self._parse_response(response)
            
            # Check for specific patterns in query
            patterns = self._check_query_patterns(query)
            
            # Override based on patterns
            if patterns['is_web_search']:
                parsed['query_type'] = 'WEB_SEARCH'
                parsed['confidence'] = 0.9
                parsed['should_retrieve'] = False
                parsed['reasoning'] = "Query requires current information"
                
            elif patterns['is_calculation']:
                parsed['query_type'] = 'CALCULATION'
                parsed['confidence'] = 0.95
                parsed['should_retrieve'] = False
                parsed['reasoning'] = "Query requires mathematical computation"
                
            elif patterns['is_direct']:
                parsed['query_type'] = 'DIRECT'
                parsed['confidence'] = 0.8
                parsed['should_retrieve'] = False
                parsed['reasoning'] = "Simple direct question"
                
            elif patterns['is_retrieval']:
                parsed['query_type'] = 'RETRIEVAL'
                parsed['confidence'] = 0.9
                parsed['should_retrieve'] = True
                parsed['reasoning'] = "Query requires document search"
            
            self.logger.info(f"Routing query '{query}' to {parsed['query_type']} "
                           f"(confidence: {parsed['confidence']})")
            
            return RouterResponse(
                query_type=QueryType(parsed['query_type'].lower()),
                confidence=parsed['confidence'],
                should_retrieve=parsed['should_retrieve'],
                retrieval_query=parsed.get('retrieval_query'),
                metadata={
                    "reasoning": parsed['reasoning'],
                    "patterns_detected": patterns
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in routing: {str(e)}")
            return RouterResponse(
                query_type=QueryType.RETRIEVAL,
                confidence=0.5,
                should_retrieve=True,
                retrieval_query=None,
                metadata={"reasoning": f"Error in routing: {str(e)}"}
            )