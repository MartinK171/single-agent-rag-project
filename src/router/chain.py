from typing import Dict, Optional
import logging
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate
from .types import QueryType, RouterResponse
from .prompts import ROUTER_PROMPT_TEMPLATE
import json

class RouterChain:
    """Chain for routing queries to appropriate handlers."""
    logging.basicConfig(level=logging.DEBUG)  # Configure logging level

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """
        Initialize the RouterChain with an LLM and a prompt template.

        :param llm: An instance of OllamaLLM, or None to use default configuration.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)

        # Initialize with the default Ollama configuration if none provided
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
        
        # Return fallback values on failure
        return {
            "query_type": "retrieval",
            "confidence": 0.5,
            "should_retrieve": True,
            "retrieval_query": "",
            "reasoning": "Failed to parse response.",
        }

    def route(self, query: str) -> RouterResponse:
        """
        Route a query to the appropriate handler by leveraging the LLM.

        :param query: The input query string.
        :return: A RouterResponse object containing the routing decision and metadata.
        """
        try:
            # Format the prompt with the provided query
            formatted_prompt = self.prompt.format(query=query)
            
            # Get the LLM's response
            self.logger.debug(f"Sending query to LLM: {query}")
            response = self.llm.invoke(formatted_prompt)
            self.logger.debug(f"Received response: {response}")
            
            # Parse the response
            parsed = self._parse_response(response)
            
            # Return a structured RouterResponse
            return RouterResponse(
                query_type=QueryType(parsed["query_type"]),
                confidence=parsed["confidence"],
                should_retrieve=parsed["should_retrieve"],
                retrieval_query=parsed.get("retrieval_query"),
                metadata={"reasoning": parsed["reasoning"]}
            )
            
        except Exception as e:
            self.logger.error(f"Error in routing: {str(e)}")
            # Return a fallback response in case of an error
            return RouterResponse(
                query_type=QueryType.RETRIEVAL,
                confidence=0.5,
                should_retrieve=True,
                retrieval_query=None,
                metadata={"reasoning": f"Error in routing: {str(e)}"}
            )
