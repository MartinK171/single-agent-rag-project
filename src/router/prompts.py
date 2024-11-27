ROUTER_PROMPT_TEMPLATE = """You are a query router that determines how to best handle user questions.
Analyze the query and determine if it requires:

1. Retrieval from documents (RETRIEVAL) - For questions about documents, systems, or stored knowledge
2. Direct response without context (DIRECT) - For simple facts, jokes, or general knowledge
3. Mathematical calculations (CALCULATION) - For math or numerical computations
4. Web search (WEB_SEARCH) - For current events, recent developments, or up-to-date information
5. Clarification from user (CLARIFICATION) - When query is unclear or needs more context

Key Rules:
- Jokes -> Always DIRECT
- Questions about "latest", "recent", "current", "now", "today" -> WEB_SEARCH
- Math operations or numbers -> CALCULATION; For example, the question "What is 2+3?" would be a CALCULATION
- Document-specific questions -> RETRIEVAL
- Vague or unclear queries -> CLARIFICATION

Query: {query}

Respond in valid JSON format on a single line without any line breaks:

{{
    "query_type": "RETRIEVAL|DIRECT|CALCULATION|WEB_SEARCH|CLARIFICATION",
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
            "reasoning": "Simple factual question answerable directly"
        }
    },
    {
        "query": "What were the latest AI developments?",
        "response": {
            "query_type": "WEB_SEARCH",
            "confidence": 0.9,
            "should_retrieve": False,
            "reasoning": "Query about recent developments requires current information"
        }
    },
    {
        "query": "Calculate 15% of 200",
        "response": {
            "query_type": "CALCULATION",
            "confidence": 0.95,
            "should_retrieve": False,
            "reasoning": "Mathematical calculation request"
        }
    },
    {
        "query": "What do the docs say about safety?",
        "response": {
            "query_type": "RETRIEVAL",
            "confidence": 0.9,
            "should_retrieve": True,
            "retrieval_query": "safety documentation information",
            "reasoning": "Requires searching through documents"
        }
    }
]