from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any

class QueryType(Enum):
   """Types of queries the router can handle."""
   RETRIEVAL = "retrieval"      # Needs context from documents 
   DIRECT = "direct"           # Can be answered directly
   CALCULATION = "calculation" # Requires mathematical computation
   WEB_SEARCH = "web_search"   # Needs web search
   CLARIFICATION = "clarification" # Needs more information

@dataclass
class RouterResponse:
   """Response from the router."""
   query_type: QueryType
   confidence: float 
   should_retrieve: bool
   metadata: Dict[str, Any]
   retrieval_query: Optional[str] = None