ROUTER_PROMPT_TEMPLATE = """You are a query router that determines how to best handle user questions.
Analyze the query and determine if it requires:

1. Retrieval from documents (RETRIEVAL)
2. Direct response without context (DIRECT)
3. Mathematical calculations (CALCULATION)
4. Clarification from user (CLARIFICATION)

Query: {query}

Respond in JSON format:
{{
    "query_type": "RETRIEVAL|DIRECT|CALCULATION|CLARIFICATION",
    "confidence": "<float between 0 and 1>",
    "should_retrieve": "<boolean>",
    "retrieval_query": "<optional: reformulated query for retrieval>",
    "reasoning": "<explanation of decision>"
}}
"""


ROUTER_EXAMPLES = [
    {
        "query": "What is the capital of France?",
        "response": {
            "query_type": "DIRECT",
            "confidence": 0.95,
            "should_retrieve": False,
            "reasoning": "This is a simple factual question that can be answered directly."
        }
    },
    {
        "query": "What does the document say about AI safety?",
        "response": {
            "query_type": "RETRIEVAL",
            "confidence": 0.9,
            "should_retrieve": True,
            "retrieval_query": "AI safety concerns and measures",
            "reasoning": "This question requires context from documents about AI safety."
        }
    }
]